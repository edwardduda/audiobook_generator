"""DramaBox prompt construction, duration budgeting, and output conditioning."""
from __future__ import annotations
import os
import re
from pathlib import Path
import numpy as np
from .core import parse_gender

SAMPLE_RATE = 48000
WORDS_PER_SEC = 2.6
DURATION_MULT = 1.1
MIN_SECONDS, MAX_SECONDS = 1.5, 35.0
NONVERBAL = re.compile(r'laugh|chuckl|giggl|sigh|sob|gasp|pause|breath|cough|sniff', re.I)
SUBJECT = re.compile(r'^(?:she|he|they|the narrator)\s+(?=[a-z])', re.I)
AGES = {'child', 'young', 'adult', 'elderly', 'unknown'}
NOUNS = {'child':('girl','boy','child'), 'young':('young woman','young man','young person'),
         'adult':('woman','man','person'), 'elderly':('elderly woman','elderly man','elderly person'),
         'unknown':('woman','man','person')}
NEUTRAL = {'', 'neutral.', 'neutral narration.', 'chapter heading.'}
DOUBLE_QUOTES = '"“”„‟«»'
# DramaBox voices prompt words that describe story content, intent, or listeners,
# so directions keep only how the voice sounds.
CONTENT_CLAUSE = re.compile(
    r'(?:,\s*|\s+)(?:(?:in order )?to (?:match|mirror|reflect|show|convey|emphasi[sz]e|capture|set|introduce|describe|build|mark)\b'
    r'|as (?:she|he|they|the|if)\b|while\b|before\b|after\b)'
    r'|,\s*(?:commanding|ordering|telling|asking|addressing|urging|describing|introducing|explaining|emphasi[sz]ing|'
    r'conveying|expressing|reflecting|matching|mirroring|setting|showing|revealing|recounting|noting|suggesting|'
    r'indicating|implying|highlighting|underscoring|acknowledging|signaling|signalling)'
    r'(?=\s)(?!\s+(?:tone|voice|edge|manner|way|quality|timbre|lilt|cadence)\b)', re.I)
CONTENT_VERB = re.compile(r'(?:describ|introduc|narrat|explain|recount|sets?\b|tell|ask|command|order|address|continu|'
                          r'conclud|summari[sz]|present|emphasi[sz]|convey|reflect)', re.I)
DELIVERY_SUBJECT = re.compile(r'(she|he|they|the narrator)\s+(\S+)\s*(.*)$', re.I)
MANNER = re.compile(r"\b(?:with|in)\s+[\w\s',-]+$|\b\w+ly$", re.I)

def clean_delivery(delivery, speaker_id='narrator'):
    """Vocal-performance part of a direction; '' when nothing performable remains."""
    text = re.sub(f'[{DOUBLE_QUOTES}]', '', ' '.join((delivery or '').split())).strip().rstrip(' .,;:')
    if text.casefold() + '.' in NEUTRAL:
        return text + '.' if text else ''
    text = CONTENT_CLAUSE.split(text, maxsplit=1)[0].strip(' ,;:')
    subject = DELIVERY_SUBJECT.match(text)
    if subject and CONTENT_VERB.match(subject.group(2)):
        manner = MANNER.search(subject.group(3))
        verb = 'reads' if speaker_id == 'narrator' else 'speaks'
        text = f'{subject.group(1)} {verb} {manner.group().strip()}' if manner else ''
    return text + '.' if text else ''

def settings():
    return {'engine':'dramabox-mlx-speech-0.5.3', 'cfg_scale':float(os.getenv('DRAMABOX_CFG_SCALE','2.5')),
            'stg_scale':float(os.getenv('DRAMABOX_STG_SCALE','1.5')), 'steps':int(os.getenv('DRAMABOX_STEPS','30')),
            'speed':float(os.getenv('DRAMABOX_SPEED','1.0')), 'seed':int(os.getenv('TTS_SEED','0')),
            'sample_rate':SAMPLE_RATE, 'trim_db':-45.0, 'duration_budget':'words-v2'}

def spoken_text(text):
    """Verbatim words with double quotes removed from the edges and softened inside.

    DramaBox speaks only what is inside its own double quotes, so any source
    double quote inside the spoken span would end the utterance early."""
    text = ' '.join(re.sub(r'\[(?:laugh|chuckle|cough)\]', ' ', text).split())
    text = text.strip(DOUBLE_QUOTES + ' ')
    return re.sub(f'[{DOUBLE_QUOTES}]', "'", text)

def _article(phrase):
    return ('An ' if phrase[:1].lower() in 'aeiou' else 'A ') + phrase

def _voice_phrase(voice):
    voice = ' '.join((voice or '').split()).strip(' .,;')
    if not voice:
        return ''
    if not re.match(r'(a|an|the)\s', voice, re.I):
        voice = ('an ' if voice[:1].lower() in 'aeiou' else 'a ') + voice
    return voice if re.search(r'\bvoice\b', voice, re.I) else voice + ' voice'

def persona(speaker_id, character=None, voice_gender='unknown'):
    character = character or {}
    gender = parse_gender(character.get('gender'))
    if gender == 'unknown':
        gender = parse_gender(voice_gender)
    index = {'female':0, 'male':1}.get(gender, 2)
    if speaker_id == 'narrator':
        noun = ('female narrator', 'male narrator', 'narrator')[index]
        voice = _voice_phrase(character.get('voice')) or 'a clear, warm, engaging voice'
    else:
        age = character.get('age') if character.get('age') in AGES else 'unknown'
        noun = NOUNS[age][index]
        voice = _voice_phrase(character.get('voice'))
    return _article(noun) + (f' with {voice}' if voice else '')

def build_prompt(row, character=None, voice_gender='unknown'):
    """One upstream-style clause: '<persona> <performance>, "<spoken>"'."""
    sid = row['speaker_id']
    delivery = clean_delivery(row.get('direction', ''), sid).rstrip(' .')
    who = persona(sid, character, voice_gender)
    spoken = spoken_text(row['dialogue'])
    if delivery.casefold() + '.' in NEUTRAL or not delivery:
        action = ('announces the chapter title clearly' if delivery.casefold() == 'chapter heading'
                  else 'reads in a steady, engaging tone' if sid == 'narrator' else 'speaks naturally')
        return f'{who} {action}, "{spoken}"'
    subject = SUBJECT.match(delivery)
    if subject:
        return f'{who} {delivery[subject.end():]}, "{spoken}"'
    return f'{who} speaks. {delivery}, "{spoken}"'

def estimate_duration(text, speed=1.0, delivery=''):
    """Seconds of speech to request.

    DramaBox fills the whole requested duration, so surplus time makes it
    improvise (repeating the line or voicing the stage direction). The upstream
    estimator's fixed 2 s of headroom is too generous for short lines."""
    words = len(re.findall(r"[\w']+", text))
    seconds = (words / WORDS_PER_SEC * DURATION_MULT + sum(text.count(p) for p in '.!?;:—') * 0.3
               + text.count(',') * 0.15 + 0.6 + (0.8 if NONVERBAL.search(delivery) else 0.0))
    return float(min(MAX_SECONDS, max(MIN_SECONDS, round(seconds / max(speed, 0.1), 2))))

def to_mono(waveform):
    data = np.asarray(waveform, dtype=np.float32)
    return data.mean(axis=0) if data.ndim == 2 else data.reshape(-1)

def trim_silence(data, sample_rate=SAMPLE_RATE, threshold_db=-45.0, pad_ms=80):
    frame = max(1, sample_rate // 100)
    usable = len(data) // frame * frame
    if not usable:
        return data
    rms = np.sqrt(np.mean(data[:usable].reshape(-1, frame) ** 2, axis=1))
    loud = np.flatnonzero(rms > 10 ** (threshold_db / 20))
    if not len(loud):
        return data
    pad = round(sample_rate * pad_ms / 1000)
    return data[max(0, loud[0] * frame - pad):min(len(data), (loud[-1] + 1) * frame + pad)]

def load_model(model_dir, text_encoder_dir):
    from .engines import hub_offline
    with hub_offline():
        from mlx_speech.generation.dramabox import DramaBoxModel
        return DramaBoxModel.from_dir(Path(model_dir), gemma_dir=Path(text_encoder_dir))

def generate(model, prompt, voice_ref, seed, config, spoken, delivery=''):
    result = model.generate(prompt, duration_s=estimate_duration(spoken, config['speed'], delivery),
                            cfg_scale=config['cfg_scale'], stg_scale=config['stg_scale'],
                            steps=config['steps'], seed=seed, voice_ref=str(voice_ref))
    data = to_mono(result.waveform)
    return trim_silence(data, int(result.sample_rate), config['trim_db']), int(result.sample_rate)
