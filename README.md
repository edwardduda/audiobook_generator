# audiobook_generator

Tools for converting PDFs into cleaned UTF-8 text and generating local, multi-voice
audiobooks with [DramaBox](https://huggingface.co/appautomaton/dramabox-tts-3.3b-bf16-mlx)
(expressive, MLX on Apple Silicon; the default) or Chatterbox.

## Setup

DramaBox's MLX runtime (`mlx-speech`) requires Python 3.13+ on an Apple Silicon Mac.
Both engines share one environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/audiobook.py models download   # ~15 GB, first time only
```

Tesseract is optional. If it is installed, pages with no usable text layer (scans) are OCR'd. Pages that already have a text layer are never OCR'd.

## Chatterbox speech on Mac

Both tools use the same Python 3.13 `.venv` and `requirements.txt` above.
Generate a short English narration (Apple Silicon GPU is selected automatically):

```bash
python src/chatterbox_tts.py "Hello! Welcome to your audiobook." -o outputs/speech.wav
afplay outputs/speech.wav
```

To use a reference voice, add `--voice path/to/reference.wav` (about 10 seconds of clean speech).
Use `--device cpu` if GPU generation fails. Model files live in the project's
`models/` folder. Missing files are downloaded from Hugging Face automatically;
when all files are present, loading is local with no Hugging Face update checks.
Audio stays local.
This command handles short passages; split books into short segments before synthesis.

Upstream: [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox).

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

## Multi-voice audiobook pipeline

The LLM is used only to turn `.txt` into a character sheet and a script. The TTS
engine chosen by `TTS_ENGINE` synthesizes speech:

- `TTS_ENGINE=dramabox` (default): Resemble AI's DramaBox, run through
  [mlx-speech](https://github.com/appautomaton/mlx-speech) on MLX. It performs a
  stage direction for every line (anger, whispers, laughter, a cracking voice) and
  clones the cast reference clip's timbre. Output is 48 kHz.
- `TTS_ENGINE=chatterbox`: Chatterbox Turbo on MPS/CUDA/CPU, 24 kHz, with only the
  `[laugh]`, `[chuckle]`, and `[cough]` reaction tags.

`script`, `synthesize`, and `run` also accept `--engine dramabox|chatterbox`.
Each engine has its own annotation prompt (`prompts/dramabox_system.txt`,
`prompts/chatterbox_system.txt`) and its own strict JSON schema for structured
outputs. A script is annotated for one engine; to switch engines, regenerate the
script in a new `--book` directory.

FFmpeg is required for the final WAV/MP3 (`brew install ffmpeg` on macOS).
Copy `.env.example` to `.env` and start the OpenAI-compatible server named there
(LM Studio by default at `http://127.0.0.1:1234/v1`, with `LLM_MODEL` loaded).
Environment variables take precedence.

### How to use

Activate the venv. Use a **new** `--book` directory for each book, and whenever
you change the source, prompt, or script-generation settings. The same directory
and command resume a failed run.

All at once:

```bash
source .venv/bin/activate
cp .env.example .env   # first time only
python src/audiobook.py run samples/audiobook_smoke.txt --book outputs/my-first-audiobook
afplay outputs/my-first-audiobook/audiobook.mp3
```

Or stage by stage (inspect `characters.json` gender and `script.jsonl` speakers
before a long synthesis):

```bash
python src/audiobook.py script path/to/book.txt --book outputs/my-book
python src/audiobook.py voices list
# Optional: write {"narrator":"catetest"} to outputs/my-book/casting.json first.
python src/audiobook.py cast --book outputs/my-book
python src/audiobook.py synthesize --book outputs/my-book
python src/audiobook.py stitch --book outputs/my-book
```

From a PDF, convert first, then pass the `.txt` to `script` or `run`. After
editing `casting.json` only, rerun `synthesize` and `stitch` — not `run` or
`script`. List and hear references with `voices list` and
`voices audition libritts_r_XXXX`.

`run` also prepares the reference library and downloads the selected engine's
weights on first use. `python src/audiobook.py models download [--engine E | --all]`
fetches them ahead of time:

- DramaBox: `models/dramabox/mlx-bf16/` (~8.5 GB) and its Gemma 3 12B text encoder
  in `models/gemma_3_12b_it_backbone/mlx-4bit/` (~6.2 GB; override with
  `DRAMABOX_GEMMA_DIR`).
- Chatterbox Turbo: `models/turbo/`.

Each folder gets a `model_manifest.json` with the resolved revision, sizes, and
checksums. Once it is present, weights load from disk with Hugging Face offline.
DramaBox needs about 17 GB of unified memory while generating. This does not
replace the original Chatterbox weights or `src/chatterbox_tts.py`. No user
interface, music, or M4B container is included.

To hear DramaBox's range, try the included scene with the `catetest` narrator:

```bash
python src/audiobook.py script samples/dramabox_expressive.txt --book outputs/lighthouse
echo '{"narrator":"catetest"}' > outputs/lighthouse/casting.json
python src/audiobook.py cast --book outputs/lighthouse
python src/audiobook.py synthesize --book outputs/lighthouse
python src/audiobook.py stitch --book outputs/lighthouse
afplay outputs/lighthouse/audiobook.mp3
```

### Character sheet

Before annotation, the LLM reads the source and writes `character_sheet.json`:
one entry per speaking character with `name`, `aliases`, `gender` (male/female/
unknown, decided from pronouns, titles, description, then conventional first
names), `gender_evidence`, `age`, `voice`, and `personality`. The annotation
pass must reuse these speaker IDs, which keeps attribution consistent. Speakers
it adds anyway are listed in `review.json`.

The sheet is yours to edit. It is reused rather than regenerated, and it takes
precedence over `characters.json`. Genders drive casting (run `cast` again for
unassigned speakers). Gender, age, and voice also form the DramaBox persona for
each line, so edits take effect on the next `synthesize`.

### DramaBox performance

Each DramaBox line is sent as one clause,
`<persona> <delivery>, "<spoken text>"`, for example:

```text
An elderly man with a deep, gravelly voice laughs bitterly, "Ha! And what would you have done with it, Nell?"
```

Only the quoted part is spoken, and it is the script `dialogue`. Double quotes
inside it become single quotes so they cannot end the utterance early.
`delivery` is the script row's `direction` (for example "He laughs bitterly"),
written by the LLM from source cues; its leading pronoun is merged into the persona.
Edit it in `script.jsonl` to change a performance; the segment is regenerated.

DramaBox may speak any prompt words that are not about the sound of the voice, so
directions are limited to vocal performance. Clauses about story content, intent,
or listeners ("to match her outburst", "as she composes herself", ", commanding
him") are removed, and content verbs become delivery verbs ("The narrator
introduces the scene with a somber tone" → "The narrator reads with a somber
tone"). This runs when the script is generated, with each change listed in
`review.json`, and again at synthesis for hand-edited scripts.

DramaBox fills the whole requested duration, and surplus time makes it repeat the
line or voice prompt words. Duration is therefore budgeted tightly from word and
punctuation counts (1.5–35 s per line, a little more when the delivery names a
laugh, sigh, or sob), and leading and trailing silence is trimmed so `stitch`
pauses control pacing. `DRAMABOX_SPEED` above 1 requests less time and packs
speech faster; below 1 risks the surplus-time problem. `DRAMABOX_CFG_SCALE`
(prompt adherence), `DRAMABOX_STG_SCALE` (clarity guidance; 0 is faster), and
`DRAMABOX_STEPS` (diffusion steps; fewer is faster) expose the model's settings.

### Script format, casting, and review

`voices build --count 20` is implied by `run` and is only needed if the library
is missing. After `script` and `cast`, inspect `script.jsonl`, `characters.json`,
`review.json`, and `casting.json`. Audition a clip with
`python src/audiobook.py voices audition libritts_r_XXXX` (an ID from
`voices list` or `casting.json`). Every stage that uses voices accepts
`--library path/to/library`.

The script contains one JSON object per line:

```json
{"position":1,"speaker_id":"narrator","dialogue":"Mara opened the door.","direction":"Neutral narration.","pause_after_ms":150}
```

`dialogue` includes both narration and character speech. Character lines omit the
source's wrapping quotation marks (`"dialogue":"Come in,"`), which are used only
to align the LLM output to the source. `direction` is never
spoken. With DramaBox it is the performance direction for the line. With
Chatterbox it is review metadata that is not sent to TTS; only the `[laugh]`,
`[chuckle]`, and `[cough]` reaction tags can be generated, and only with matching
source cues. Narration, actions, and phrases such as “she said” belong to the
narrator. Uncertain speakers fall back to the narrator and appear in `review.json`.

The editable `prompts/<engine>_system.txt` and `prompts/character_sheet_system.txt`
define LLM behavior (`--prompt` and `--sheet-prompt` override them). Generated text is
aligned to the source; altered, repeated, missing, or reordered passages are rejected.
Whitespace and curly/straight quotation differences are accepted, but final text is
reconstructed from the source (minus the wrapping quotes on character lines). These checks protect textual fidelity, not perfect
character attribution or TTS pronunciation. Review the script before a long synthesis.

`casting.json` maps book character IDs to distinct library IDs:

```json
{"narrator":"libritts_r_1272","mara":"libritts_r_1462"}
```

These IDs are examples: use voices present in your actual manifest. Assignments are
deterministic and persist across reruns. Edit this file to change voices, then rerun
`synthesize` and `stitch`. Existing assignments are preserved by `cast`; duplicate
voice assignments are rejected. New assignments prefer a LibriTTS `male`/`female`
reference matching the character `gender` written by script generation. `unknown`
and unlabeled clips such as custom WAVs can receive any remaining voice. List labels
with `python src/audiobook.py voices list`.

### Library downloads and expansion

```bash
python src/audiobook.py voices build --count 40 --split dev.clean --max-rows 10000
python src/audiobook.py voices build --count 100 --split train.clean.100 --max-rows 50000
```

The count is the desired total library size. Existing voices are retained. The
builder streams the selected split, stops once enough speakers qualify or the row
limit is reached, and retains complete 6–15-second non-silent utterances. It prefers
clips closest to 10 seconds among candidates encountered for each speaker. Streaming
still transfers Parquet data and may read many utterances before reaching new speakers.
A partial library is saved when the scan limit is reached; increase the limit or use
another split. `--revision` can pin a Hugging Face dataset revision.

Every library clip has its utterance/speaker ID, source transcript, resolved dataset
revision, split, duration, checksum, and attribution in `manifest.json`. References
are mono 24 kHz WAVs. Raw audio bytes are decoded with SoundFile, without TorchCodec.
Dataset: [mythicinfinity/libritts_r](https://huggingface.co/datasets/mythicinfinity/libritts_r),
[LibriTTS-R](https://www.openslr.org/141/), CC BY 4.0. Keep the supplied attribution
when distributing reference material. Chatterbox output carries Resemble's Perth
watermark; the MLX DramaBox port does not add one. DramaBox weights are under the
LTX-2 Community License.

### Configuration and recovery

- Hosted LLMs must support `/chat/completions`; change `LLM_BASE_URL`, `LLM_API_KEY`,
  and `LLM_MODEL`. Use `LLM_RESPONSE_FORMAT=json_schema`, `json_object`, or `text`
  depending on the provider. No native Anthropic/Google adapters are included.
- `LLM_REASONING_EFFORT` is sent as `reasoning_effort` when set. `.env.example`
  uses `none` because LM Studio otherwise routes Qwen 3.x JSON into
  `reasoning_content`; clear it for providers or models that reject the field.
- `LLM_CONTEXT_TOKENS` must reflect the context actually loaded by the server.
  `LLM_MAX_OUTPUT_TOKENS` includes any model reasoning tokens. Context/output
  truncation reduces source chunk size; malformed or unfaithful responses are
  retried twice. Persistent errors stop the pipeline with files in `diagnostics/`.
- `script` and `run` accept `--chunk-chars`, `--segment-chars`, and `--prompt`.
  `--chunk-chars 0` (the default) sends the whole remaining source in one LLM
  request so speakers stay in memory. Use a positive value (>=256) only if the
  model cannot return the full chapter. Completed chunks live in
  `checkpoints/script.json`. A changed source, prompt, or generation
  configuration requires a new output directory. Corrected service failures can
  resume with the same command.
- Editing the final `script.jsonl` is supported; rerunning `script` preserves it.
  Keep positions contiguous starting at 1. Run `cast` if you introduce a new speaker.
  Manual script edits are your responsibility for source fidelity.
- Chatterbox: `TTS_DEVICE=auto` selects CUDA, MPS, then CPU; `--device cpu` is
  available for unsupported GPU operations. `TTS_TEMPERATURE` controls sampling.
  DramaBox always runs on MLX. `TTS_SEED` applies to both engines. Seeds improve
  repeatability but do not guarantee identical audio across devices.
- Synthesis fingerprints spoken text, reference audio, weights, engine, and
  generation settings; for DramaBox it fingerprints the full prompt, so persona or
  direction edits regenerate that line. It reuses only matching valid WAVs. Pauses
  (and, for Chatterbox, directions) do not trigger TTS work. Keep the segment
  manifest with the WAVs.
- `stitch` rejects missing, corrupt, or stale segments and ignores unrelated WAVs.
  It concatenates in script position order, adding the specified pauses without
  crossfades. Defaults are 150 ms between segments, 350 ms at paragraph endings,
  and 900 ms at chapter boundaries. FFmpeg streams PCM into an RF64-capable WAV
  and creates a 128 kbps MP3. It never loads the whole audiobook into RAM.
- Final files are `audiobook.wav` and `audiobook.mp3`; `audio_manifest.json` records
  checksums and expected duration. FFmpeg errors are saved in `ffmpeg.log`.
- `.env`, downloaded weights, library audio, and generated books are ignored by Git.
  API keys are not written to checkpoints. Use `AUDIOBOOK_DEBUG=1` for tracebacks.

Run `python -m pytest -q` for PDF regression tests plus mocked pipeline, streaming,
casting, resumption, and real FFmpeg assembly tests. The supplied smoke passage is
original and safe to use with either local or hosted LLM endpoints.
