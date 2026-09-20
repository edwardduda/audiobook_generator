#!/usr/bin/env python3
"""Convert a book PDF into cleaned UTF-8 text for LLM parsing."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz

LOGGER = logging.getLogger("pdf_to_text")

HEADER_FOOTER_LINES = 2
RUNNING_LINE_MIN_PAGES = 3
RUNNING_LINE_FRACTION = 0.4
PAGE_NUMBER_RE = re.compile(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$")
HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
SENTENCE_END_RE = re.compile(r'[.!?"”\']$')
HEADING_RE = re.compile(
    r"^(chapter|book|prologue|epilogue|part|glossary|map|contents|foreword|preface)\b",
    re.IGNORECASE,
)


@dataclass
class PageText:
    number: int
    text: str


@dataclass
class ExtractionResult:
    pages: list[PageText]
    total_pages: int
    skipped_image_pages: list[tuple[int, int]] = field(default_factory=list)


def extract_pages(doc: fitz.Document) -> ExtractionResult:
    """Extract selectable text; skip empty image-only pages with a warning."""
    pages: list[PageText] = []
    skipped: list[tuple[int, int]] = []

    for index, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        if not text.strip():
            images = page.get_images()
            if images:
                skipped.append((index, len(images)))
                LOGGER.warning(
                    "Skipping image-only page %s (%s image%s)",
                    index,
                    len(images),
                    "s" if len(images) != 1 else "",
                )
            continue
        pages.append(PageText(number=index, text=text))

    return ExtractionResult(
        pages=pages,
        total_pages=len(doc),
        skipped_image_pages=skipped,
    )


def _edge_lines(text: str) -> tuple[list[str], list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return [], []
    headers = lines[:HEADER_FOOTER_LINES]
    footers = lines[-HEADER_FOOTER_LINES:] if len(lines) > HEADER_FOOTER_LINES else []
    return headers, footers


def detect_running_lines(pages: list[PageText]) -> set[str]:
    """Find repeating header/footer lines and page-number-like lines."""
    if len(pages) < RUNNING_LINE_MIN_PAGES:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        headers, footers = _edge_lines(page.text)
        seen: set[str] = set()
        for line in headers + footers:
            normalized = PAGE_NUMBER_RE.sub("<page>", line)
            if normalized not in seen:
                counts[normalized] += 1
                seen.add(normalized)

    threshold = max(RUNNING_LINE_MIN_PAGES, int(len(pages) * RUNNING_LINE_FRACTION))
    running = {line for line, count in counts.items() if count >= threshold}
    return running


def _is_running_line(line: str, running: set[str]) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if PAGE_NUMBER_RE.match(stripped):
        return True
    normalized = PAGE_NUMBER_RE.sub("<page>", stripped)
    return stripped in running or normalized in running


def strip_running_lines(text: str, running: set[str]) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        if _is_running_line(line, running):
            continue
        kept.append(line)
    return "\n".join(kept)


def _is_heading(line: str) -> bool:
    if HEADING_RE.match(line):
        return True
    letters = [c for c in line if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters) and len(line) <= 60


def _should_join_lines(current: str, nxt: str) -> bool:
    if _is_heading(current) or _is_heading(nxt):
        return False
    if nxt[:1].islower():
        return True
    return not SENTENCE_END_RE.search(current)


def reconstruct_paragraphs(text: str) -> str:
    """Rejoin hyphenated wraps and turn PDF line breaks into paragraphs."""
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    blocks = re.split(r"\n\s*\n", text)
    paragraphs: list[str] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        merged: list[str] = []
        current = lines[0]
        for line in lines[1:]:
            if _should_join_lines(current, line):
                current = f"{current} {line}"
            else:
                merged.append(current)
                current = line
        merged.append(current)
        paragraphs.extend(merged)

    return "\n\n".join(paragraphs).strip()


def convert(
    pdf_path: Path,
    txt_path: Path,
    *,
    keep_page_breaks: bool = False,
) -> ExtractionResult:
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF: {pdf_path}\n{exc}") from exc

    with doc:
        result = extract_pages(doc)

    if not result.pages:
        raise RuntimeError(
            f"No extractable text in {pdf_path}. "
            "This often means the PDF is scanned/image-only (OCR is not used)."
        )

    running = detect_running_lines(result.pages)
    cleaned_pages: list[str] = []
    for page in result.pages:
        stripped = strip_running_lines(page.text, running)
        body = reconstruct_paragraphs(stripped)
        if not body:
            continue
        if keep_page_breaks:
            cleaned_pages.append(f"--- page {page.number} ---\n\n{body}")
        else:
            cleaned_pages.append(body)

    if not any(part.strip() for part in cleaned_pages):
        raise RuntimeError(f"No extractable text remained after cleaning {pdf_path}.")

    skipped = ", ".join(
        f"{page} ({count} image{'s' if count != 1 else ''})"
        for page, count in result.skipped_image_pages
    ) or "none"

    header = (
        f"# source: {pdf_path.name}\n"
        f"# pages: {result.total_pages}\n"
        f"# skipped_image_pages: {skipped}\n\n"
    )
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(header + "\n\n".join(cleaned_pages) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cleaned UTF-8 text from a book PDF for LLM parsing."
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
        result = convert(pdf_path, txt_path, keep_page_breaks=args.keep_page_breaks)
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
