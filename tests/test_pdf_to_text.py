from pathlib import Path

import fitz
import pytest

from pdf_to_text import convert, join_wrapped_lines, main, resolve_hyphen_break


def _write_pdf(path: Path, build) -> Path:
    doc = fitz.open()
    build(doc)
    doc.save(path)
    doc.close()
    return path


def test_resolve_hyphen_keeps_compound_adjectives():
    assert resolve_hyphen_break("earth", "colored") == "earth-colored"
    assert resolve_hyphen_break("black", "cloaked") == "black-cloaked"
    assert resolve_hyphen_break("gath", "ered") == "gathered"


def test_join_wrapped_lines_does_not_turn_hard_hyphens_into_spaces():
    text = join_wrapped_lines(["cloud-capped peaks", "were visible."])
    assert "cloud-capped" in text
    assert "cloud capped" not in text


def test_title_and_drop_cap_paragraph_are_separate(tmp_path: Path):
    def build(doc: fitz.Document) -> None:
        page = doc.new_page(width=595, height=842)
        page.insert_text((250, 80), "CHAPTER", fontsize=20)
        page.insert_text((290, 110), "1", fontsize=20)
        page.insert_text((220, 210), "An Empty Road", fontsize=20)
        writer = fitz.TextWriter(page.rect)
        writer.append((72, 300), "T", fontsize=24)
        writer.append((90, 308), "he Wheel of Time turns, and Ages come and pass.", fontsize=14)
        writer.write_text(page)
        page.insert_text((72, 360), "Born below the ever cloud-capped peaks.", fontsize=14)

    pdf = _write_pdf(tmp_path / "chapter.pdf", build)
    out = tmp_path / "chapter.txt"
    convert(pdf, out, ocr=False, plain=True)
    text = out.read_text(encoding="utf-8")
    assert "An Empty Road\n\nThe Wheel of Time turns" in text
    assert "An Empty Road The Wheel" not in text
    assert "CHAPTER 1" in text


def test_hyphenation_on_page(tmp_path: Path):
    def build(doc: fitz.Document) -> None:
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(72, 80, 280, 400),
            "Gusts whipped the earth-\ncolored wool. They gath-\nered near the green. A black-\ncloaked rider waited.",
            fontsize=14,
        )

    pdf = _write_pdf(tmp_path / "hyphens.pdf", build)
    out = tmp_path / "hyphens.txt"
    convert(pdf, out, ocr=False, plain=True)
    text = out.read_text(encoding="utf-8")
    assert "earth-colored" in text
    assert "earthcolored" not in text
    assert "black-cloaked" in text
    assert "blackcloaked" not in text
    assert "gathered" in text


def test_running_header_and_page_numbers_stripped(tmp_path: Path):
    def build(doc: fitz.Document) -> None:
        for i in range(4):
            page = doc.new_page()
            page.insert_text((72, 24), "The Eye of the World", fontsize=9)
            page.insert_text((72, 80), f"Unique body paragraph number {i} continues with enough letters to keep.", fontsize=14)
            page.insert_text((300, 770), str(i + 1), fontsize=9)

    pdf = _write_pdf(tmp_path / "headers.pdf", build)
    out = tmp_path / "headers.txt"
    convert(pdf, out, ocr=False, plain=True)
    text = out.read_text(encoding="utf-8")
    assert "The Eye of the World" not in text
    assert "Unique body paragraph number 0" in text
    assert "\n1\n" not in text


def test_ornament_does_not_skip_text_page(tmp_path: Path):
    def build(doc: fitz.Document) -> None:
        page = doc.new_page()
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 16), False)
        pix.clear_with(120)
        page.insert_image(fitz.Rect(250, 80, 310, 130), pixmap=pix)
        page.insert_text((72, 200), "The villagers gathered near the green after the storm passed over.", fontsize=14)

    pdf = _write_pdf(tmp_path / "ornament.pdf", build)
    out = tmp_path / "ornament.txt"
    result = convert(pdf, out, ocr=False, plain=True)
    assert result.skipped_image_pages == []
    assert "villagers gathered" in out.read_text(encoding="utf-8")


def test_image_only_page_skipped(tmp_path: Path):
    def build(doc: fitz.Document) -> None:
        page = doc.new_page()
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 64, 64), False)
        pix.clear_with(10)
        page.insert_image(page.rect, pixmap=pix)

    pdf = _write_pdf(tmp_path / "map.pdf", build)
    out = tmp_path / "map.txt"
    with pytest.raises(RuntimeError, match="No extractable text"):
        convert(pdf, out, ocr=False, plain=True)


def test_two_column_reading_order(tmp_path: Path):
    def build(doc: fitz.Document) -> None:
        page = doc.new_page()
        left = "Alpha paragraph lives in the left column and must be read first in order."
        right = "Beta paragraph lives in the right column and must be read second in order."
        for i in range(3):
            page.insert_textbox(fitz.Rect(40, 80 + i * 90, 280, 160 + i * 90), f"{left} {i}.", fontsize=12)
            page.insert_textbox(fitz.Rect(320, 80 + i * 90, 560, 160 + i * 90), f"{right} {i}.", fontsize=12)

    pdf = _write_pdf(tmp_path / "columns.pdf", build)
    out = tmp_path / "columns.txt"
    convert(pdf, out, ocr=False, plain=True)
    text = out.read_text(encoding="utf-8")
    assert text.find("Alpha paragraph") < text.find("Beta paragraph")
    assert text.rfind("Alpha paragraph") < text.find("Beta paragraph") or text.find("left column and must be read first in order. 2") < text.find("Beta paragraph lives in the right column and must be read second in order. 0")


def test_footnotes_are_omitted(tmp_path: Path):
    def build(doc: fitz.Document) -> None:
        page = doc.new_page()
        page.insert_text((72, 80), "Rand walked toward the inn with his bow already nocked.", fontsize=14)
        page.insert_text((72, 790), "1 This footnote must not be spoken.", fontsize=7)

    pdf = _write_pdf(tmp_path / "notes.pdf", build)
    out = tmp_path / "notes.txt"
    convert(pdf, out, ocr=False, plain=True)
    text = out.read_text(encoding="utf-8")
    assert "Rand walked toward the inn" in text
    assert "footnote must not be spoken" not in text


def test_cross_page_wrap(tmp_path: Path):
    def build(doc: fitz.Document) -> None:
        page = doc.new_page()
        page.insert_text((72, 80), "Only trees that kept leaf or needle through the winter had any green about", fontsize=14)
        page = doc.new_page()
        page.insert_text((72, 80), "them. Snarls of last year's bramble spread brown webs over stone.", fontsize=14)

    pdf = _write_pdf(tmp_path / "wrap.pdf", build)
    out = tmp_path / "wrap.txt"
    convert(pdf, out, ocr=False, plain=True)
    text = out.read_text(encoding="utf-8")
    assert "green about them." in text
    assert "about\n\nthem" not in text


def test_missing_pdf_exits_nonzero(tmp_path: Path):
    assert main([str(tmp_path / "missing.pdf")]) == 1
