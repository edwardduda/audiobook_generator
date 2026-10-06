"""Shared file contracts and source-preserving text utilities."""
from __future__ import annotations
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

TAGS = {'laugh', 'chuckle', 'cough'}
ROOT = Path(__file__).resolve().parents[2]
GENDERS = {'male', 'female', 'unknown'}

class PipelineError(Exception):
    pass

def parse_gender(value):
    if value is None or value == '':
        return 'unknown'
    if not isinstance(value, str):
        raise PipelineError('Gender must be male, female, or unknown.')
    mapped = {'f':'female','female':'female','m':'male','male':'male','unknown':'unknown'}
    key = value.strip().casefold()
    if key not in mapped:
        raise PipelineError('Gender must be male, female, or unknown.')
    return mapped[key]

def digest(value):
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def save_json(path, obj):
    atomic_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def clean_source(path):
    lines = [line for line in Path(path).read_text(encoding='utf-8-sig').splitlines()
             if not line.lstrip().startswith('#') and not re.fullmatch(r'\s*--- page \d+ ---\s*', line)]
    text = '\n'.join(lines).strip()
    if not text:
        raise PipelineError('No spoken source text remains after removing metadata.')
    return text

def normalize(text):
    return ''.join(normalized_chars(text)[0])

def normalized_chars(text):
    """Whitespace-free comparison plus offsets back into untouched source."""
    chars, offsets = [], []
    table = str.maketrans({'“':'"', '”':'"', '‘':"'", '’':"'"})
    for i, char in enumerate(text):
        if not char.isspace():
            chars.append(char.translate(table))
            offsets.append(i)
    return chars, offsets

def split_text(text, limit=300):
    if limit < 32:
        raise PipelineError('Segment/chunk limit must be at least 32 characters.')
    result = []
    while len(text) > limit:
        matches = list(re.finditer(r'[.!?;:,][”"\']?\s+|\s+', text[:limit + 1]))
        if not matches:
            raise PipelineError('An unbroken source token exceeds the segment limit.')
        preferred = [m for m in matches if m.start() >= limit // 3 and text[m.start()] in '.!?;:,']
        cut = (preferred or matches)[-1].end()
        result.append(text[:cut].strip())
        text = text[cut:].strip()
    if text.strip():
        result.append(text.strip())
    return result

def book_characters(book):
    """characters.json overlaid by the user-editable character_sheet.json."""
    merged = {}
    for name in ('characters.json', 'character_sheet.json'):
        path = Path(book)/name
        data = read_json(path) if path.is_file() else {}
        if not isinstance(data, dict):
            raise PipelineError(f'{name} must map speaker IDs to character objects.')
        for sid, item in data.items():
            if not isinstance(item, dict):
                continue
            current = merged.setdefault(sid, {})
            for key, value in item.items():
                if key == 'gender' and parse_gender(value) == 'unknown' and current.get('gender'):
                    continue
                if value not in ('', None, []):
                    current[key] = value
            current['gender'] = parse_gender(current.get('gender'))
    return merged

def load_script(book):
    path = Path(book) / 'script.jsonl'
    rows = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            row = json.loads(line)
            assert type(row['position']) is int and row['position'] == line_no
            assert isinstance(row['speaker_id'], str) and re.fullmatch(r'[a-z][a-z0-9_]*', row['speaker_id'])
            assert isinstance(row['dialogue'], str) and row['dialogue'].strip()
            assert all(tag in TAGS for tag in re.findall(r'\[([^\]\n]+)\]', row['dialogue'])), 'Unsupported inline TTS tag'
            assert isinstance(row['direction'], str)
            assert type(row['pause_after_ms']) is int and 0 <= row['pause_after_ms'] <= 10000
        except (ValueError, KeyError, TypeError, AssertionError) as e:
            raise PipelineError(f'Invalid script row {line_no}: {e}') from e
        rows.append(row)
    if not rows:
        raise PipelineError('Script is empty.')
    return rows
