"""Bounded LibriTTS-R reference library and persistent book casting."""
from __future__ import annotations
import io
import re
import subprocess
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from .core import PipelineError, ROOT, book_characters, file_hash, load_script, parse_gender, read_json, save_json

DATASET = 'mythicinfinity/libritts_r'
ATTRIBUTION = 'LibriTTS-R, Koizumi et al. (2023), via mythicinfinity/libritts_r; CC BY 4.0. https://www.openslr.org/141/ https://creativecommons.org/licenses/by/4.0/'

SPEAKER_LABELS = ROOT / 'data' / 'libritts_speakers.json'
_SPEAKER_GENDERS = None

def speaker_genders():
    global _SPEAKER_GENDERS
    if _SPEAKER_GENDERS is None:
        _SPEAKER_GENDERS = {}
        if SPEAKER_LABELS.is_file():
            _SPEAKER_GENDERS = {str(speaker_id): parse_gender(sex) for speaker_id, sex in read_json(SPEAKER_LABELS).items()}
    return _SPEAKER_GENDERS

def apply_voice_genders(library, manifest, persist=True):
    labels = speaker_genders()
    changed = False
    for voice in manifest['voices'].values():
        try:
            current = parse_gender(voice.get('gender'))
        except PipelineError:
            current = 'unknown'
        if current != 'unknown':
            if voice.get('gender') != current:
                voice['gender'] = current
                changed = True
            continue
        labeled = labels.get(str(voice.get('speaker_id')), 'unknown')
        if voice.get('gender') != labeled:
            voice['gender'] = labeled
            changed = True
    if persist and changed:
        save_json(Path(library)/'manifest.json', manifest)
    return manifest

def load_library(path):
    path = Path(path)
    manifest = read_json(path/'manifest.json')
    if manifest.get('dataset') != DATASET or not isinstance(manifest.get('voices'),dict):
        raise PipelineError('Invalid voice library manifest.')
    return apply_voice_genders(path, manifest)

def resolve_reference(library, voice):
    root = Path(library).resolve()
    path = (root/voice['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise PipelineError(f'Invalid or missing reference clip: {path}')
    if file_hash(path) != voice['sha256']:
        raise PipelineError(f'Reference checksum changed: {path}. Rebuild or restore the library clip.')
    info = sf.info(path)
    if info.frames / info.samplerate <= 5 or info.channels != 1:
        raise PipelineError(f'Reference must be mono and longer than five seconds: {path}')
    return path

def decode_candidate(row):
    audio = row['audio']
    raw = audio.get('bytes')
    if raw is None:
        # Dataset paths are often machine-specific strings; only actual files are usable.
        path = Path(audio.get('path') or '')
        if not path.is_file():
            return None
        raw = path.read_bytes()
    try:
        data, sr = sf.read(io.BytesIO(raw),dtype='float32',always_2d=True)
    except (RuntimeError, ValueError):
        return None
    if not len(data) or not np.isfinite(data).all():
        return None
    data = data.mean(axis=1)
    duration = len(data)/sr
    if not 6 <= duration <= 15 or float(np.sqrt(np.mean(data**2))) < 0.003 or np.mean(np.abs(data)>=0.999) > 0.01:
        return None
    if sr != 24000:
        import librosa
        data = librosa.resample(data,orig_sr=sr,target_sr=24000)
    return data, duration

def build_library(library, count=20, split='dev.clean', revision='main', max_rows=10000, rows=None):
    library = Path(library)
    if count < 1 or max_rows < 1:
        raise PipelineError('Voice count and scan limit must be positive.')
    manifest_path = library/'manifest.json'
    manifest = load_library(library) if manifest_path.exists() else {'version':1,'dataset':DATASET,'attribution':ATTRIBUTION,'voices':{}}
    for voice in manifest['voices'].values():
        resolve_reference(library,voice)
    if len(manifest['voices']) >= count:
        print(f"Voice library ready: {len(manifest['voices'])} voices",flush=True)
        return manifest
    if rows is None:
        from datasets import Audio, load_dataset
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(DATASET,revision=revision).sha
        config = 'clean' if '.clean' in split else 'other'
        rows = load_dataset(DATASET,config,split=split,streaming=True,revision=revision).cast_column('audio',Audio(decode=False))
    candidates = {}
    scanned = 0
    for row in rows:
        scanned += 1
        if scanned > max_rows:
            break
        speaker = str(row['speaker_id'])
        if not re.fullmatch(r'[0-9]+',speaker):
            continue
        voice_id = 'libritts_r_' + speaker
        if voice_id in manifest['voices']:
            continue
        candidate = decode_candidate(row)
        if candidate is None:
            continue
        data,duration = candidate
        if voice_id not in candidates or abs(duration-10) < abs(candidates[voice_id][2]-10):
            candidates[voice_id] = (row,data,duration)
        if len(manifest['voices']) + len(candidates) >= count:
            break
    library.mkdir(parents=True,exist_ok=True)
    labels = speaker_genders()
    for voice_id,(row,data,duration) in sorted(candidates.items()):
        relative = f'clips/{voice_id}.wav'
        path = library/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_suffix('.tmp.wav')
        sf.write(temp,data,24000,subtype='PCM_16')
        temp.replace(path)
        manifest['voices'][voice_id] = {'path':relative,'speaker_id':str(row['speaker_id']),
            'utterance_id':str(row['id']), 'transcript':row.get('text_original') or row.get('text_normalized',''),
            'duration':len(data)/24000, 'sample_rate':24000, 'sha256':file_hash(path),
            'gender':labels.get(str(row['speaker_id']),'unknown'),
            'dataset':DATASET,'revision':revision,'split':split,'attribution':ATTRIBUTION}
        save_json(manifest_path,manifest)
    if len(manifest['voices']) < count:
        raise PipelineError(f"Saved {len(manifest['voices'])}/{count} voices after scanning {min(scanned,max_rows)} rows. Increase --max-rows or expand with --split train.clean.100.")
    print(f"Voice library ready: {len(manifest['voices'])} voices in {library}",flush=True)
    return manifest

def character_genders(book):
    return {sid: item['gender'] for sid, item in book_characters(book).items()}

def pick_voice(available, voices, needed):
    available = list(available)
    if needed in {'male','female'}:
        matching = [voice_id for voice_id in available if parse_gender(voices[voice_id].get('gender')) == needed]
        if matching:
            return matching[0]
        print(f'No {needed} library voice left; assigning an unmatched reference.',flush=True)
    return available[0]

def cast_book(book,library):
    book = Path(book)
    rows = load_script(book)
    manifest = load_library(library)
    path = book/'casting.json'
    casting = read_json(path) if path.exists() else {}
    if not isinstance(casting,dict) or any(not isinstance(v,str) for v in casting.values()):
        raise PipelineError('casting.json must map character IDs to library voice IDs.')
    speakers = sorted({row['speaker_id'] for row in rows},key=lambda x:(x != 'narrator',x))
    for sid,voice_id in casting.items():
        if voice_id not in manifest['voices']:
            raise PipelineError(f'Unknown cast voice {voice_id} for {sid}.')
    if len(set(casting.values())) != len(casting):
        raise PipelineError('Casting must assign distinct voices to each character.')
    available = sorted(set(manifest['voices'])-set(casting.values()))
    missing = [sid for sid in speakers if sid not in casting]
    if len(missing) > len(available):
        raise PipelineError(f'Need {len(missing)} additional distinct voices but only {len(available)} are available. Expand with voices build --count N.')
    genders = character_genders(book)
    gendered = [sid for sid in missing if genders.get(sid,'unknown') in {'male','female'}]
    other = [sid for sid in missing if sid not in gendered]
    for sid in gendered + other:
        voice_id = pick_voice(available, manifest['voices'], genders.get(sid, 'unknown'))
        casting[sid] = voice_id
        available.remove(voice_id)
    save_json(path,casting)
    print(f'Casting ready: {path}',flush=True)
    return casting

def list_voices(library):
    manifest = load_library(library)
    print(f"{'voice_id':<22} {'gender':<8} speaker",flush=True)
    for voice_id, voice in sorted(manifest['voices'].items()):
        print(f"{voice_id:<22} {parse_gender(voice.get('gender')):<8} {voice.get('speaker_id','')}",flush=True)

def audition(library, voice_id):
    manifest = load_library(library)
    if voice_id not in manifest['voices']:
        raise PipelineError(f'Unknown voice {voice_id}.')
    voice = manifest['voices'][voice_id]
    path = resolve_reference(library,voice)
    print(f"{voice_id}: {voice['transcript']}\n{path}",flush=True)
    player = shutil.which('afplay') or shutil.which('ffplay')
    if not player:
        raise PipelineError(f'No audio player found. Open {path} manually.')
    args = [player,str(path)] if Path(player).name == 'afplay' else [player,'-nodisp','-autoexit',str(path)]
    subprocess.run(args,check=True)
