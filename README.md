# audiobook_generator

Tools for turning books into audiobook scripts. This repo currently converts a PDF into cleaned UTF-8 text that an LLM or TTS engine can read.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tesseract is optional. If it is installed, pages with no usable text layer (scans) are OCR'd. Pages that already have a text layer are never OCR'd.

## PDF to text

```bash
python src/pdf_to_text.py path/to/book.pdf
python src/pdf_to_text.py path/to/book.pdf -o path/to/book.txt
python src/pdf_to_text.py path/to/book.pdf --plain
python src/pdf_to_text.py path/to/book.pdf --keep-page-breaks
python src/pdf_to_text.py path/to/book.pdf --no-ocr
```

Example files live in `samples/`.

Default output is the same directory and name as the PDF, with a `.txt` suffix.

The converter:

- Reads layout blocks (not raw line soup), so chapter titles stay off the first paragraph
- Attaches a drop cap to the first body paragraph (`T` + `he` → `The`)
- Rejoins line wraps; keeps compounds such as `earth-colored` / `black-cloaked` and only closes hyphenation when the glued form is a real word (`gathered`)
- Ignores images; small ornaments do not drop a text page; large image-only pages are skipped
- Strips running headers/footers, page numbers, footnotes, and sidenotes
- Reads two-column pages left column then right
- Joins paragraphs split across a page break
- OCR-falls back only when extractable text is empty or garbage

## Output contract (TTS-ready)

The `.txt` is UTF-8. Spoken content is:

- Headings on their own lines (`CHAPTER 1`, then `An Empty Road`)
- Body as paragraphs separated by a blank line; each paragraph is a single line
- No page numbers, running headers, footnotes, or image captions
- `#` lines at the top are metadata; TTS and later pipeline steps should ignore them

`--plain` writes only spoken text (no `#` header). `--keep-page-breaks` inserts `--- page N ---` markers for debugging and is not for TTS.

Example:

```
# source: book.pdf
# pages: 16
# skipped_image_pages: none
# ocr_pages: none
# tts: skip lines starting with #

CHAPTER 1

An Empty Road

The Wheel of Time turns, and Ages come and pass...
```

The process exits with a non-zero status if the PDF cannot be opened or yields no extractable text.

## Tests

```bash
python -m pytest
```

Tests build tiny synthetic PDFs. Do not commit copyrighted books.

## Images and scanned PDFs

Embedded images are not parsed as pictures. They will not crash the script.

If a drop cap is an *image* instead of a letter, that character is still missing unless OCR can read it. Glyph drop caps in the text layer are merged into the first paragraph.

Fully scanned PDFs use OCR when Tesseract is available; otherwise the script warns and skips those pages.
