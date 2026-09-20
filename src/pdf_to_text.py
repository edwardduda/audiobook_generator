#!/usr/bin/env python3
"""Convert a book PDF into cleaned UTF-8 text for LLM parsing and TTS."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import fitz

LOGGER = logging.getLogger("pdf_to_text")

HEADER_FRACTION = 0.08
FOOTER_FRACTION = 0.92
FOOTNOTE_FRACTION = 0.80
RUNNING_LINE_MIN_PAGES = 3
RUNNING_LINE_FRACTION = 0.4
MIN_BODY_LETTERS = 25
LARGE_IMAGE_AREA_FRACTION = 0.25
OCR_DPI = 200
ZIPF_CLOSED_WORD = 3.5

PAGE_NUMBER_RE = re.compile(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$")
HEADING_RE = re.compile(
    r"^(chapter|book|prologue|epilogue|part|glossary|map|contents|foreword|preface)\b",
    re.IGNORECASE,
)
SENTENCE_END_RE = re.compile(r'[.!?"”\']["”\']?$')
ELLIPSIS_RE = re.compile(r"(?:\.\s*){3,}|…")
COMPOUND_TAILS = frozenset(
    {
        "colored",
        "coloured",
        "cloaked",
        "clad",
        "haired",
        "skinned",
        "eyed",
        "handed",
        "headed",
        "looking",
        "faced",
        "voiced",
        "born",
        "made",
        "strewn",
        "capped",
        "filled",
        "stained",
        "soaked",
        "bound",
        "worn",
        "like",
        "shaped",
        "sized",
        "covered",
        "washed",
        "kissed",
        "struck",
        "ridden",
        "blooded",
        "tempered",
    }
)
CHAPTER_NUMBER_RE = re.compile(r"^\d+$")


@dataclass
class Segment:
    kind: str  # heading | body
    text: str
    page: int


@dataclass
class PageText:
    number: int
    text: str
    segments: list[Segment] = field(default_factory=list)


@dataclass
class ExtractionResult:
    pages: list[PageText]
    total_pages: int
    skipped_image_pages: list[tuple[int, int]] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)


def _zipf(word: str) -> float:
    try:
        from wordfreq import zipf_frequency

        return float(zipf_frequency(word.lower(), "en"))
    except Exception:
        return 0.0


@lru_cache(maxsize=1)
def _system_words() -> set[str]:
    path = Path("/usr/share/dict/words")
    if not path.is_file():
        return set()
    return {line.strip().lower() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()}


def _is_closed_word(word: str) -> bool:
    if _zipf(word) >= ZIPF_CLOSED_WORD:
        return True
    return word.lower() in _system_words()


def resolve_hyphen_break(left: str, right: str) -> str:
    """Join a line-end hyphen. Keep English compounds; close real wrapped words."""
    closed = f"{left}{right}"
    hyphenated = f"{left}-{right}"
    if right.lower() in COMPOUND_TAILS:
        return hyphenated
    closed_ok = _is_closed_word(closed)
    hyphen_ok = _zipf(hyphenated) >= 2.5 or hyphenated.lower() in _system_words()
    if closed_ok and (not hyphen_ok or _zipf(closed) >= _zipf(hyphenated) + 0.5):
        return closed
    return hyphenated


def join_wrapped_lines(lines: list[str]) -> str:
    pieces: list[str] = []
    current = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not current:
            current = line
            continue
        if len(current) <= 2 and current.isalpha() and line[:1].islower():
            current = current + line
            continue
        if current.endswith("-"):
            stem = current[:-1]
            match_left = re.search(r"([A-Za-z']+)$", stem)
            match_right = re.match(r"([A-Za-z']+)(.*)$", line)
            if match_left and match_right:
                prefix = stem[: match_left.start(1)]
                current = (
                    prefix
                    + resolve_hyphen_break(match_left.group(1), match_right.group(1))
                    + match_right.group(2)
                )
            else:
                current = stem + line
        else:
            current = f"{current} {line}"
    if current:
        pieces.append(current)
    return " ".join(pieces).strip()


def _spans_to_text(spans: list[dict]) -> str:
    if not spans:
        return ""
    parts = [spans[0]["text"]]
    for prev, span in zip(spans, spans[1:]):
        text = span["text"]
        prev_text = prev["text"]
        gap = span["bbox"][0] - prev["bbox"][2]
        prev_size = float(prev.get("size") or 0)
        size = float(span.get("size") or 0)
        next_body = text.lstrip()
        drop_cap = (
            len(prev_text.strip()) <= 2
            and prev_text.strip().isalpha()
            and bool(next_body[:1].islower())
            and prev_size > size * 1.1
        )
        if drop_cap:
            parts.append(next_body)
        elif prev_text.endswith(" ") or text.startswith(" ") or gap < 1.0:
            parts.append(text)
        else:
            parts.append(" " + text)
    return "".join(parts)


def _block_lines(block: dict) -> list[str]:
    lines: list[str] = []
    for line in block.get("lines") or []:
        text = _spans_to_text(line.get("spans") or []).strip()
        if text:
            lines.append(text)
    return lines


def _block_font_size(block: dict) -> float:
    sizes: list[float] = []
    for line in block.get("lines") or []:
        for span in line.get("spans") or []:
            sizes.append(float(span.get("size") or 0))
    return sum(sizes) / len(sizes) if sizes else 0.0


def _modal_body_size(blocks: list[dict]) -> float:
    sizes: list[int] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        text_len = sum(len(span.get("text") or "") for line in block.get("lines") or [] for span in line.get("spans") or [])
        size = _block_font_size(block)
        if text_len >= 40 and size:
            sizes.append(round(size * 2) / 2)
    if not sizes:
        return 12.0
    return Counter(sizes).most_common(1)[0][0]


def _image_area_fraction(page: fitz.Page, blocks: list[dict]) -> float:
    page_area = abs(page.rect.width * page.rect.height) or 1.0
    area = 0.0
    for block in blocks:
        if block.get("type") == 1:
            x0, y0, x1, y1 = block["bbox"]
            area += abs((x1 - x0) * (y1 - y0))
    return area / page_area


def _two_column_order(text_blocks: list[dict], page: fitz.Page) -> list[dict]:
    width = page.rect.width
    if len(text_blocks) < 6:
        return sorted(text_blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    full: list[dict] = []
    columns: list[dict] = []
    for block in text_blocks:
        x0, _, x1, _ = block["bbox"]
        if (x1 - x0) >= width * 0.55:
            full.append(block)
        else:
            columns.append(block)

    left = [b for b in columns if (b["bbox"][0] + b["bbox"][2]) / 2 < width / 2]
    right = [b for b in columns if b not in left]
    if len(left) < 3 or len(right) < 3:
        return sorted(text_blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    left.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    right.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    full.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    col_top = min(b["bbox"][1] for b in columns)
    before = [b for b in full if b["bbox"][1] < col_top - 4]
    after = [b for b in full if b not in before]
    return before + left + right + after


def _is_heading_block(text: str, size: float, body_size: float, width_frac: float) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if HEADING_RE.match(stripped) or CHAPTER_NUMBER_RE.match(stripped):
        return True
    letters = [c for c in stripped if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and len(stripped) <= 60:
        return True
    words = stripped.split()
    title_case = 1 <= len(words) <= 8 and stripped[0].isupper() and not stripped.endswith(".")
    larger = size >= body_size * 1.15
    short_centered = len(stripped) <= 80 and width_frac < 0.7 and title_case
    return larger and (title_case or len(stripped) <= 80) or (larger and short_centered)


def _classify_block(
    block: dict,
    page: fitz.Page,
    body_size: float,
    *,
    footnote: bool,
) -> str | None:
    lines = _block_lines(block)
    if not lines:
        return None
    text = join_wrapped_lines(lines)
    if not text:
        return None
    x0, y0, x1, y1 = block["bbox"]
    height = page.rect.height or 1.0
    width = page.rect.width or 1.0
    size = _block_font_size(block)
    if y1 <= height * HEADER_FRACTION and len(text) <= 80:
        return "header"
    if y0 >= height * FOOTER_FRACTION and (PAGE_NUMBER_RE.match(text) or len(text) <= 80):
        return "footer"
    if footnote and y0 >= height * FOOTNOTE_FRACTION and size < body_size * 0.85:
        return "footnote"
    if size < body_size * 0.8 and (x0 > width * 0.72 or x1 < width * 0.28) and len(text) <= 200:
        return "sidenote"
    if _is_heading_block(text, size, body_size, (x1 - x0) / width):
        return "heading"
    return "body"


def _ocr_page(page: fitz.Page) -> str:
    try:
        textpage = page.get_textpage_ocr(language="eng", dpi=OCR_DPI)
        return page.get_text("text", textpage=textpage) or ""
    except Exception as exc:
        LOGGER.warning("OCR failed on page %s: %s", page.number + 1, exc)
        return ""


def _garbage_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    letters = sum(ch.isalpha() for ch in stripped)
    if letters < MIN_BODY_LETTERS:
        return True
    return letters / max(len(stripped), 1) < 0.35


def _segments_from_dict(page: fitz.Page, page_number: int) -> list[Segment]:
    data = page.get_text("dict")
    blocks = data.get("blocks") or []
    text_blocks = [b for b in blocks if b.get("type") == 0]
    body_size = _modal_body_size(text_blocks)
    ordered = _two_column_order(text_blocks, page)
    segments: list[Segment] = []
    pending_heading: list[str] = []

    def flush_heading() -> None:
        nonlocal pending_heading
        if not pending_heading:
            return
        text = " ".join(pending_heading).strip()
        if text:
            segments.append(Segment("heading", text, page_number))
        pending_heading = []

    for block in ordered:
        kind = _classify_block(block, page, body_size, footnote=True)
        if kind in {None, "header", "footer", "footnote", "sidenote"}:
            continue
        text = join_wrapped_lines(_block_lines(block))
        if kind == "heading":
            if CHAPTER_NUMBER_RE.match(text) and pending_heading:
                pending_heading.append(text)
            elif pending_heading and HEADING_RE.match(pending_heading[0]) and CHAPTER_NUMBER_RE.match(text):
                pending_heading.append(text)
            else:
                flush_heading()
                pending_heading = [text]
            continue
        flush_heading()
        segments.append(Segment("body", text, page_number))
    flush_heading()
    return segments


def extract_pages(doc: fitz.Document, *, ocr: bool = True) -> ExtractionResult:
    """Extract selectable text; skip empty image-heavy pages; OCR only if needed."""
    pages: list[PageText] = []
    skipped: list[tuple[int, int]] = []
    ocr_pages: list[int] = []

    for index, page in enumerate(doc, start=1):
        data = page.get_text("dict")
        blocks = data.get("blocks") or []
        images = [b for b in blocks if b.get("type") == 1]
        segments = _segments_from_dict(page, index)
        raw = page.get_text("text") or ""
        used_ocr = False

        if ocr and _garbage_text(" ".join(s.text for s in segments) or raw):
            ocr_text = _ocr_page(page)
            if ocr_text.strip() and not _garbage_text(ocr_text):
                used_ocr = True
                ocr_pages.append(index)
                segments = [
                    Segment("body", join_wrapped_lines(line.strip() for line in para.splitlines() if line.strip()), index)
                    for para in re.split(r"\n\s*\n", ocr_text)
                    if para.strip()
                ]

        body_letters = sum(ch.isalpha() for s in segments for ch in s.text)
        if body_letters < MIN_BODY_LETTERS:
            image_frac = _image_area_fraction(page, blocks)
            if images and (image_frac >= LARGE_IMAGE_AREA_FRACTION or not segments):
                skipped.append((index, len(page.get_images())))
                LOGGER.warning(
                    "Skipping image-only page %s (%s image%s)",
                    index,
                    len(page.get_images()),
                    "s" if len(page.get_images()) != 1 else "",
                )
                continue
            if not segments:
                continue

        text = "\n\n".join(s.text for s in segments)
        pages.append(PageText(number=index, text=text, segments=segments))
        if used_ocr:
            LOGGER.info("Used OCR on page %s", index)

    return ExtractionResult(
        pages=pages,
        total_pages=len(doc),
        skipped_image_pages=skipped,
        ocr_pages=ocr_pages,
    )


def detect_running_lines(pages: list[PageText]) -> set[str]:
    """Find repeating header/footer lines and page-number-like lines."""
    if len(pages) < RUNNING_LINE_MIN_PAGES:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        if not lines:
            continue
        seen: set[str] = set()
        edges = lines[:2] + (lines[-2:] if len(lines) > 2 else [])
        for line in edges:
            normalized = PAGE_NUMBER_RE.sub("<page>", line)
            if normalized not in seen:
                counts[normalized] += 1
                seen.add(normalized)

    threshold = max(RUNNING_LINE_MIN_PAGES, int(len(pages) * RUNNING_LINE_FRACTION))
    return {line for line, count in counts.items() if count >= threshold}


def _is_running_segment(segment: Segment, running: set[str]) -> bool:
    stripped = segment.text.strip()
    if PAGE_NUMBER_RE.match(stripped):
        return True
    normalized = PAGE_NUMBER_RE.sub("<page>", stripped)
    return stripped in running or normalized in running


def _is_continuation(previous: str, nxt: str) -> bool:
    prev = previous.rstrip()
    first = nxt.lstrip()
    if not prev or not first:
        return False
    if first[:1].islower():
        return True
    if prev.endswith("-"):
        return True
    return not SENTENCE_END_RE.search(prev)


def stitch_segments(
    pages: list[PageText],
    running: set[str],
    *,
    keep_page_breaks: bool,
) -> list[str]:
    stitched: list[Segment] = []
    for page in pages:
        segs = [s for s in page.segments if not _is_running_segment(s, running)]
        if keep_page_breaks:
            if stitched:
                stitched.append(Segment("heading", f"--- page {page.number} ---", page.number))
            stitched.extend(segs)
            continue
        if stitched and segs:
            prev, first = stitched[-1], segs[0]
            if prev.kind == "body" and first.kind == "body" and _is_continuation(prev.text, first.text):
                if prev.text.endswith("-"):
                    joined = join_wrapped_lines([prev.text, first.text])
                else:
                    joined = f"{prev.text.rstrip()} {first.text.lstrip()}"
                stitched[-1] = Segment("body", joined, prev.page)
                segs = segs[1:]
        stitched.extend(segs)

    lines: list[str] = []
    for segment in stitched:
        text = normalize_tts(segment.text)
        if text:
            lines.append(text)
    return lines


def normalize_tts(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = ELLIPSIS_RE.sub("...", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def convert(
    pdf_path: Path,
    txt_path: Path,
    *,
    keep_page_breaks: bool = False,
    ocr: bool = True,
    plain: bool = False,
) -> ExtractionResult:
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF: {pdf_path}\n{exc}") from exc

    with doc:
        result = extract_pages(doc, ocr=ocr)

    if not result.pages:
        raise RuntimeError(
            f"No extractable text in {pdf_path}. "
            "This often means the PDF is scanned/image-only and OCR produced nothing."
        )

    running = detect_running_lines(result.pages)
    paragraphs = stitch_segments(result.pages, running, keep_page_breaks=keep_page_breaks)
    if not paragraphs:
        raise RuntimeError(f"No extractable text remained after cleaning {pdf_path}.")

    skipped = ", ".join(
        f"{page} ({count} image{'s' if count != 1 else ''})"
        for page, count in result.skipped_image_pages
    ) or "none"
    ocr_note = ", ".join(str(p) for p in result.ocr_pages) or "none"

    body = "\n\n".join(paragraphs) + "\n"
    if not plain:
        header = (
            f"# source: {pdf_path.name}\n"
            f"# pages: {result.total_pages}\n"
            f"# skipped_image_pages: {skipped}\n"
            f"# ocr_pages: {ocr_note}\n"
            "# tts: skip lines starting with #\n\n"
        )
        body = header + body

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(body, encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cleaned UTF-8 text from a book PDF for LLM parsing and TTS."
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .txt path (default: same name as the PDF, .txt suffix)",
    )
    parser.add_argument(
        "--keep-page-breaks",
        action="store_true",
        help="Insert --- page N --- markers between pages",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Do not OCR pages that have empty or garbage text",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Write only spoken text (no # metadata header)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    pdf_path = args.pdf.expanduser().resolve()
    if not pdf_path.is_file():
        LOGGER.error("PDF not found: %s", pdf_path)
        return 1

    txt_path = args.output
    if txt_path is None:
        txt_path = pdf_path.with_suffix(".txt")
    else:
        txt_path = txt_path.expanduser().resolve()

    try:
        result = convert(
            pdf_path,
            txt_path,
            keep_page_breaks=args.keep_page_breaks,
            ocr=not args.no_ocr,
            plain=args.plain,
        )
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info(
        "Wrote %s (%s of %s pages with text, %s image-only pages skipped)",
        txt_path,
        len(result.pages),
        result.total_pages,
        len(result.skipped_image_pages),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
