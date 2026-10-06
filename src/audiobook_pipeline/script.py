"""OpenAI-compatible annotation, strict source alignment, and chunk checkpoints."""
from __future__ import annotations
import copy
import json
import os
import re
import time
from pathlib import Path
from .core import (ROOT, TAGS, GENDERS, PipelineError, atomic_text, clean_source, digest,
                   normalize, normalized_chars, parse_gender, read_json, save_json, split_text)
from .dramabox import DOUBLE_QUOTES, clean_delivery
from .engines import engine_name

class Truncated(PipelineError):
    pass

def obj(properties):
    return {'type':'object', 'properties':properties, 'required':list(properties), 'additionalProperties':False}

STRING = {'type':'string'}
PRONOUNS = {'he', 'she', 'they', 'him', 'her', 'them', 'his', 'hers', 'their', 'i', 'you', 'we', 'it'}
HEADING = re.compile(r'\s*((?:chapter|book|part|prologue|epilogue)\b[^\n]{0,110})(?:\n|$)', re.I)
AGES = ['adult', 'child', 'elderly', 'unknown', 'young']
PROFILE_FIELDS = ('gender_evidence', 'age', 'voice', 'personality')
CHARACTER = obj({'speaker_id':STRING, 'name':STRING, 'aliases':{'type':'array','items':STRING},
                 'gender':{'type':'string','enum':sorted(GENDERS)}})
SCHEMAS = {
    'chatterbox': obj({'characters':{'type':'array', 'items':CHARACTER},
        'segments':{'type':'array', 'items':obj({'text':STRING, 'speaker_id':STRING,
            'direction':STRING, 'reaction':{'type':'string','enum':['', *sorted(TAGS)]},
            'uncertain':{'type':'boolean'}})}}),
    'dramabox': obj({'characters':{'type':'array', 'items':CHARACTER},
        'segments':{'type':'array', 'items':obj({'text':STRING, 'speaker_id':STRING,
            'delivery':STRING, 'uncertain':{'type':'boolean'}})}}),
}
SCHEMA = SCHEMAS['chatterbox']
SHEET_SCHEMA = obj({'characters':{'type':'array', 'items':obj({'speaker_id':STRING, 'name':STRING,
    'aliases':{'type':'array','items':STRING}, 'gender':{'type':'string','enum':sorted(GENDERS)},
    'gender_evidence':STRING, 'age':{'type':'string','enum':AGES}, 'voice':STRING, 'personality':STRING})}})

def default_prompt(engine):
    return ROOT/'prompts'/f'{engine}_system.txt'

SHEET_PROMPT = ROOT/'prompts/character_sheet_system.txt'

class LLMClient:
    def __init__(self):
        import httpx
        self.base_url = os.getenv('LLM_BASE_URL', 'http://127.0.0.1:1234/v1').rstrip('/')
        self.model = os.getenv('LLM_MODEL', 'the-crow-9b-creative-writing-opus4.6-distill-heretic')
        self.mode = os.getenv('LLM_RESPONSE_FORMAT', 'json_schema')
        if self.mode not in {'json_schema','json_object','text'}:
            raise PipelineError('LLM_RESPONSE_FORMAT must be json_schema, json_object, or text.')
        self.context_tokens = int(os.getenv('LLM_CONTEXT_TOKENS', '10240'))
        self.max_output = int(os.getenv('LLM_MAX_OUTPUT_TOKENS', '4096'))
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.1'))
        self.reasoning_effort = os.getenv('LLM_REASONING_EFFORT', '').strip()
        key = os.getenv('LLM_API_KEY', 'lm-studio')
        self.http = httpx.Client(timeout=float(os.getenv('LLM_TIMEOUT_SECONDS', '180')),
                                 headers={'Authorization':f'Bearer {key}'} if key else {})
        self.identity = {'url':self.base_url, 'model':self.model, 'format':self.mode,
                         'context':self.context_tokens, 'output':self.max_output, 'temperature':self.temperature}
        if self.reasoning_effort:
            self.identity['reasoning_effort'] = self.reasoning_effort

    def close(self):
        self.http.close()

    def annotate(self, prompt, data, feedback='', schema=('audiobook_annotations', SCHEMA)):
        import httpx
        messages = [{'role':'system','content':prompt}, {'role':'user','content':json.dumps(data, ensure_ascii=False)}]
        if feedback:
            messages.append({'role':'user', 'content':'Previous attempt rejected. Return a complete corrected result. ' + feedback[:1500]})
        # A conservative English-text estimate, plus safety space for schema/chat framing.
        estimated = len(json.dumps(messages, ensure_ascii=False).encode()) / 2 + self.max_output + 800
        if estimated > self.context_tokens:
            raise Truncated('Request would exceed configured context budget; reduce chunk size or increase loaded context.')
        payload = {'model':self.model, 'messages':messages, 'temperature':self.temperature,
                   'max_tokens':self.max_output, 'stream':False}
        if self.reasoning_effort:
            payload['reasoning_effort'] = self.reasoning_effort
        if self.mode == 'json_schema':
            payload['response_format'] = {'type':'json_schema','json_schema':{'name':schema[0],'strict':True,'schema':schema[1]}}
        elif self.mode == 'json_object':
            payload['response_format'] = {'type':'json_object'}
        for attempt in range(3):
            try:
                response = self.http.post(self.base_url + '/chat/completions', json=payload)
            except httpx.TransportError as e:
                if attempt == 2:
                    raise PipelineError(f'LLM connection failed ({type(e).__name__}); check endpoint and timeout.') from e
                time.sleep(2 ** attempt)
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            if response.status_code >= 400:
                message = response.text[:1500]
                if response.status_code in (400, 413) and re.search(r'context|token.*limit|too long', message, re.I):
                    raise Truncated('Provider rejected request context length.')
                raise PipelineError(f'LLM HTTP {response.status_code}: {message}')
            try:
                choice = response.json()['choices'][0]
                if choice.get('finish_reason') == 'length':
                    raise Truncated('LLM output was truncated.')
                if choice.get('finish_reason') != 'stop':
                    raise PipelineError(f"LLM did not finish normally: {choice.get('finish_reason')}")
                message = choice['message']
                # Some servers (LM Studio with Qwen 3.x) route schema-constrained output to reasoning_content.
                content = (message.get('content') or '').strip() or (message.get('reasoning_content') or '').strip()
                content = re.sub(r'^<think>.*?</think>\s*', '', content, flags=re.S)
                return json.loads(content)
            except (KeyError, IndexError, TypeError, ValueError) as e:
                raise PipelineError('LLM returned invalid JSON or an invalid completion envelope.') from e
        raise PipelineError('LLM request failed.')

PARAGRAPH_BREAK = re.compile(r'\n[ \t]*\n')

def quoted_mask(text):
    """Conservative direct-speech mask.

    An unclosed quote ends at the paragraph break or at the next opening quote, so
    one missing mark (common in extracted PDFs) cannot invert the rest of the text."""
    mask = [False] * len(text)
    start = None
    closing = None
    for i, c in enumerate(text):
        prev = text[i-1] if i else ' '
        nxt = text[i+1] if i + 1 < len(text) else ' '
        opens = c == '“' or c == '"' and (prev.isspace() or prev in '([{—–-') and not nxt.isspace()
        if start is not None and (c == '\n' and PARAGRAPH_BREAK.match(text, i)
                                  or opens and closing in ('"', '”') and (c == '“' or closing == '"')):
            mask[start:i] = [True] * (i - start)
            start, closing = None, None
        if start is None:
            if opens or c == '‘':
                start, closing = i, {'“':'”', '"':'"', '‘':'’'}[c]
            elif c == "'" and not prev.isalnum():
                start, closing = i, "'"
        elif c == closing:
            if c in ("'", '’') and nxt.isalnum():
                continue
            mask[start:i+1] = [True] * (i + 1 - start)
            start, closing = None, None
    return mask

def merge_characters(registry, updates):
    registry = copy.deepcopy(registry)
    aliases = {}
    for sid, character in registry.items():
        character.setdefault('gender', 'unknown')
        for name in [sid, character['name'], *character['aliases']]:
            aliases[name.casefold()] = sid
    remap = {}
    for item in updates:
        if not isinstance(item, dict) or not isinstance(item.get('name'), str) or not item['name'].strip():
            raise PipelineError('Invalid character name.')
        sid = item.get('speaker_id')
        names = item.get('aliases')
        if not isinstance(sid, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', sid) or not isinstance(names,list) or any(not isinstance(n,str) for n in names):
            raise PipelineError('Invalid character ID or aliases.')
        if sid in PRONOUNS:
            raise PipelineError('Use a named character ID, not a pronoun; use uncertain narrator when identity is unknown.')
        names = [name for name in names if name.strip().casefold() not in PRONOUNS]
        candidates = {aliases[n.casefold()] for n in [sid,item['name'],*names] if n.casefold() in aliases}
        if len(candidates) > 1:
            raise PipelineError('Character aliases collide with multiple existing speakers.')
        canonical = next(iter(candidates), sid)
        if canonical == 'narrator' and sid != 'narrator':
            raise PipelineError('A character cannot alias the narrator.')
        remap[sid] = canonical
        existing = registry.setdefault(canonical, {'name':item['name'], 'aliases':[], 'gender':'unknown'})
        existing.setdefault('gender', 'unknown')
        existing['aliases'] = sorted(set(existing['aliases'] + names + ([item['name'],sid] if sid != canonical else [])))
        gender = parse_gender(item.get('gender'))
        if existing['gender'] == 'unknown' and gender != 'unknown':
            existing['gender'] = gender
        for field in PROFILE_FIELDS:
            value = item.get(field)
            if value is None:
                continue
            if not isinstance(value, str) or field == 'age' and value not in AGES:
                raise PipelineError(f'Invalid character {field}.')
            if value.strip() and existing.get(field, 'unknown') in ('', 'unknown'):
                existing[field] = value.strip()
        for name in [sid, item['name'], *names]:
            aliases[name.casefold()] = canonical
    return registry, remap

def _span_pause(source, offsets, start, end):
    start_offset, end_offset = offsets[start], offsets[end - 1] + 1
    original = source[start_offset:end_offset]
    boundary = source[end_offset: offsets[end] if end < len(offsets) else len(source)]
    next_text = source[offsets[end]:] if end < len(offsets) else ''
    heading = bool(re.match(r'^(chapter|part|book|prologue|epilogue)\b', original.strip(), re.I)) and '\n' not in original and len(original) < 120
    pause = 900 if heading or re.match(r'^(chapter|part|book|prologue|epilogue)\b', next_text, re.I) else 350 if '\n\n' in boundary or end == len(offsets) else 150
    return original, start_offset, end_offset, pause

def _emit_span(aligned, warnings, source, offsets, mask, start, end, sid, direction, reaction, recovered=False):
    original, start_offset, end_offset, pause = _span_pause(source, offsets, start, end)
    if recovered:
        warnings.append({'text': original, 'reason': 'Recovered omitted source as narrator.'})
        sid = 'narrator'
        direction = direction or 'Neutral narration.'
        reaction = ''
    mixed = sid != 'narrator' and any(not mask[i] for i in range(start_offset, end_offset) if source[i].isalnum())
    if mixed:
        warnings.append({'text': original, 'reason': 'Split character speech from glued attribution.'})
        run_start = start
        quoted = mask[offsets[start]]
        for index in range(start, end + 1):
            at_end = index == end
            now_quoted = False if at_end else mask[offsets[index]]
            if at_end or now_quoted != quoted:
                if index > run_start:
                    piece, _, _, piece_pause = _span_pause(source, offsets, run_start, index)
                    if piece.strip():
                        aligned.append({'speaker_id': sid if quoted else 'narrator', 'text': piece,
                                        'direction': direction if quoted else 'Neutral narration.',
                                        'reaction': reaction if quoted else '', 'pause_after_ms': piece_pause})
                run_start = index
                quoted = now_quoted
        return end
    if sid == 'narrator' and any(mask[start_offset:end_offset]):
        warnings.append({'text': original, 'reason': 'Quoted speech assigned to narrator; review speaker attribution.'})
    aligned.append({'speaker_id': sid, 'text': original, 'direction': direction, 'reaction': reaction, 'pause_after_ms': pause})
    return end

def validate_annotations(source, result, registry, source_mask=None, engine='chatterbox'):
    if not isinstance(result,dict) or not isinstance(result.get('characters'),list) or not isinstance(result.get('segments'),list) or not result['segments']:
        raise PipelineError('Response must contain characters and a nonempty segments array.')
    updated, remap = merge_characters(registry, result['characters'])
    chars, offsets = normalized_chars(source)
    expected = ''.join(chars)
    mask = quoted_mask(source) if source_mask is None else source_mask
    cursor, aligned, warnings = 0, [], []
    fields = ('text','speaker_id','delivery') if engine == 'dramabox' else ('text','speaker_id','direction','reaction')
    for segment in result['segments']:
        if not isinstance(segment,dict) or any(not isinstance(segment.get(k),str) for k in fields) or type(segment.get('uncertain')) is not bool:
            raise PipelineError('Invalid segment fields.')
        if engine == 'dramabox':
            delivery = clean_delivery(segment['delivery'], segment['speaker_id'])
            if normalize(delivery).rstrip('.') != normalize(segment['delivery']).rstrip('.'):
                warnings.append({'text': segment['text'], 'reason': f"Delivery trimmed to vocal performance: {segment['delivery']!r} -> {delivery!r}"})
            neutral = 'Neutral narration.' if segment['speaker_id'] == 'narrator' else 'Neutral.'
            segment = {**segment, 'direction':delivery or neutral, 'reaction':''}
        part = normalize(segment['text'])
        if not part:
            raise PipelineError(f'Source fidelity mismatch at character {cursor}; expected {expected[cursor:cursor+80]!r}. Preserve every source word in order.')
        if not expected.startswith(part, cursor):
            found = expected.find(part, cursor) if len(part) >= 8 else -1
            if found < 0:
                raise PipelineError(f'Source fidelity mismatch at character {cursor}; expected {expected[cursor:cursor+80]!r}. Preserve every source word in order.')
            cursor = _emit_span(aligned, warnings, source, offsets, mask, cursor, found, 'narrator', 'Neutral narration.', '', recovered=True)
        end = cursor + len(part)
        sid = remap.get(segment['speaker_id'], segment['speaker_id'])
        if segment['uncertain']:
            sid = 'narrator'
            warnings.append({'text': segment['text'], 'reason': 'Uncertain speaker; used narrator.'})
        if sid not in updated:
            raise PipelineError(f'Unknown speaker ID: {sid}')
        reaction = segment['reaction']
        if reaction not in TAGS | {''}:
            raise PipelineError(f'Unsupported Turbo reaction: {reaction}')
        if reaction:
            start_offset, end_offset = offsets[cursor], offsets[end - 1] + 1
            nearby = source[max(0, start_offset - 150):min(len(source), end_offset + 150)]
            roots = {'laugh': r'\blaugh\w*\b', 'chuckle': r'\bchuckl\w*\b', 'cough': r'\bcough\w*\b'}
            if not re.search(roots[reaction], nearby, re.I):
                raise PipelineError(f'Reaction {reaction} is not supported by nearby source text.')
        cursor = _emit_span(aligned, warnings, source, offsets, mask, cursor, end, sid, segment['direction'], reaction)
    if cursor != len(expected):
        if len(expected) - cursor >= 8 or aligned:
            cursor = _emit_span(aligned, warnings, source, offsets, mask, cursor, len(expected), 'narrator', 'Neutral narration.', '', recovered=True)
        else:
            raise PipelineError(f'Source fidelity mismatch: omitted {len(expected)-cursor} trailing characters.')
    return aligned, updated, warnings

def choose_end(source,start,limit):
    end = min(len(source),start+limit)
    if end == len(source):
        return end
    cut = source.rfind('\n\n',start+limit//3,end)
    if cut >= 0:
        return cut + 2
    matches = list(re.finditer(r'[.!?][”"\']?\s+|\s+',source[start:end]))
    if not matches:
        raise PipelineError('Cannot chunk an unbroken source token.')
    return start + matches[-1].end()

NARRATOR = {'name':'Narrator','aliases':[],'gender':'unknown'}

def build_character_sheet(source, book, client, prompt, chunk_chars=0):
    """LLM pass 1: a persistent, user-editable cast list used to map speakers and voices."""
    book = Path(book)
    path = book/'character_sheet.json'
    if path.exists():
        sheet = read_json(path)
        if not isinstance(sheet, dict) or any(not isinstance(v, dict) for v in sheet.values()):
            raise PipelineError('character_sheet.json must map speaker IDs to character objects.')
        return sheet
    identity = digest({'version':1,'source':source,'prompt':prompt,'client':client.identity,'chunk':chunk_chars})
    checkpoint = book/'checkpoints/character_sheet.json'
    state = {'identity':identity,'offset':0,'characters':{'narrator':copy.deepcopy(NARRATOR)}}
    if checkpoint.exists() and read_json(checkpoint).get('identity') == identity:
        state = read_json(checkpoint)
    limit = len(source) if chunk_chars == 0 else chunk_chars
    offset = state['offset']
    while offset < len(source):
        end = choose_end(source, offset, limit)
        chunk = source[offset:end]
        print(f'Building character sheet from source characters {offset + 1}–{end} / {len(source)}', flush=True)
        feedback, shrink = '', False
        for attempt in range(3):
            result = None
            try:
                result = client.annotate(prompt, {'source':chunk, 'known_characters':state['characters']},
                                         feedback, schema=('character_sheet', SHEET_SCHEMA))
                if not isinstance(result, dict) or not isinstance(result.get('characters'), list):
                    raise PipelineError('Response must contain a characters array.')
                characters, _ = merge_characters(state['characters'], result['characters'])
                break
            except Truncated as e:
                save_json(book/'diagnostics'/f'sheet_{offset}_attempt_{attempt}.json', {'error':str(e), 'response':result})
                if min(limit, len(chunk)) <= 256:
                    raise PipelineError('Context/output budget insufficient for the character sheet at minimum chunk size.') from e
                limit = max(256, min(limit, len(chunk)) // 2)
                shrink = True
                break
            except PipelineError as e:
                feedback = str(e)
                save_json(book/'diagnostics'/f'sheet_{offset}_attempt_{attempt}.json', {'error':feedback, 'response':result})
                if attempt == 2:
                    raise PipelineError(f'Character sheet failed validation after three attempts: {e}. See diagnostics.') from e
        if shrink:
            continue
        state.update(characters=characters, offset=end)
        save_json(checkpoint, state)
        offset = end
    save_json(path, state['characters'])
    print(f'Character sheet ready: {path} ({len(state["characters"]) - 1} characters)', flush=True)
    return state['characters']

def generate_script(input_path, book, client, chunk_chars=0, segment_chars=300, prompt_path=None,
                    engine=None, sheet_prompt_path=None):
    book = Path(book)
    engine = engine_name(engine)
    source = clean_source(input_path)
    source_mask = quoted_mask(source)
    prompt = Path(prompt_path or default_prompt(engine)).read_text(encoding='utf-8')
    sheet_prompt = Path(sheet_prompt_path or SHEET_PROMPT).read_text(encoding='utf-8')
    identity = digest({'version':2,'engine':engine,'source':source,'prompt':prompt,'sheet_prompt':sheet_prompt,
                       'client':client.identity,'chunk':chunk_chars,'segment':segment_chars})
    checkpoint = book/'checkpoints/script.json'
    if checkpoint.exists() and read_json(checkpoint).get('identity') != identity:
        raise PipelineError('Source, engine, or script configuration changed. Use a new book output directory to preserve the previous run.')
    sheet = build_character_sheet(source, book, client, sheet_prompt, chunk_chars)
    state = {'identity':identity,'offset':0,'characters':{'narrator':copy.deepcopy(NARRATOR), **copy.deepcopy(sheet)},
             'segments':[],'warnings':[]}
    if checkpoint.exists():
        state = read_json(checkpoint)
    schema = ('audiobook_annotations', SCHEMAS[engine])
    offset = state['offset']
    current_limit = len(source) if chunk_chars == 0 else chunk_chars
    while offset < len(source):
        heading = HEADING.match(source, offset)
        if heading:
            if state['segments']:
                state['segments'][-1]['pause_after_ms'] = 900
            state['segments'].append({'speaker_id':'narrator', 'text':heading.group(1),
                'direction':'Chapter heading.', 'reaction':'', 'pause_after_ms':900})
            state['offset'] = heading.end()
            save_json(checkpoint, state)
            offset = heading.end()
            continue
        end = choose_end(source,offset,current_limit)
        # Headings are rendered deterministically, never entrusted to the LLM.
        next_heading = re.search(r'\n\n(?=(?:chapter|book|part|prologue|epilogue)\b)', source[offset:end], re.I)
        if next_heading:
            end = offset + next_heading.end()
        chunk = source[offset:end]
        if not chunk.strip():
            state['offset'] = end
            save_json(checkpoint,state)
            offset = end
            continue
        span = f'{offset + 1}–{end} / {len(source)}'
        print(f'Annotating entire source ({span})' if chunk_chars == 0 and end == len(source) else f'Annotating source characters {span}',flush=True)
        feedback = ''
        shrink = False
        for attempt in range(3):
            prior = [{'speaker_id':item['speaker_id'],'text':item['text'][-240:]} for item in state['segments'][-24:]]
            data = {'source':chunk,'context_before':source[max(0,offset-500):offset],
                    'context_after':source[end:end+500], 'characters':state['characters'],
                    'prior_segments':prior}
            result = None
            try:
                result = client.annotate(prompt,data,feedback,schema=schema)
                aligned, registry, warnings = validate_annotations(chunk,result,state['characters'],source_mask[offset:end],engine)
                warnings += [{'text':registry[sid]['name'], 'reason':'Speaker not in character_sheet.json; added during annotation.'}
                             for sid in sorted(set(registry) - set(state['characters']) - set(sheet))]
                break
            except Truncated as e:
                save_json(book/'diagnostics'/f'chunk_{offset}_attempt_{attempt}.json',{'error':str(e),'response':result})
                if min(current_limit, len(chunk)) <= 256:
                    raise PipelineError('Context/output budget still insufficient at minimum chunk size. Increase context/output budget or use another model.') from e
                current_limit = max(256,min(current_limit,len(chunk))//2)
                shrink = True
                break
            except PipelineError as e:
                feedback = str(e)
                save_json(book/'diagnostics'/f'chunk_{offset}_attempt_{attempt}.json',{'error':feedback,'response':result})
                if attempt == 2:
                    raise PipelineError(f'Chunk at offset {offset} failed fidelity/format validation after three attempts: {e}. See diagnostics; completed chunks remain resumable.') from e
        if shrink:
            continue
        state['segments'].extend(aligned)
        state['characters'] = registry
        state['warnings'].extend(warnings)
        state['offset'] = end
        save_json(checkpoint,state)
        offset = end
    rows = []
    for segment in state['segments']:
        prefix = f"[{segment['reaction']}] " if segment['reaction'] else ''
        paragraphs = re.split(r'\n\s*\n', segment['text'])
        for paragraph_number, paragraph in enumerate(paragraphs):
            pieces = split_text(paragraph,segment_chars-len(prefix))
            if segment['speaker_id'] != 'narrator':
                # Source quote marks are kept for alignment but are not part of the spoken line.
                pieces = [p for p in (piece.strip(DOUBLE_QUOTES + ' ') for piece in pieces) if p] or pieces
            for i,piece in enumerate(pieces):
                pause = 150
                if i == len(pieces)-1:
                    pause = segment['pause_after_ms'] if paragraph_number == len(paragraphs)-1 else 350
                rows.append({'position':len(rows)+1,'speaker_id':segment['speaker_id'],
                             'dialogue':(prefix if i == 0 and paragraph_number == 0 else '')+piece,
                             'direction':segment['direction'], 'pause_after_ms':pause})
    # A completed checkpoint is authoritative; preserve a user's edited final script on rerun.
    script_path = book/'script.jsonl'
    if not script_path.exists():
        atomic_text(script_path,''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    save_json(book/'characters.json',state['characters'])
    save_json(book/'review.json',{'warnings':state['warnings'],'source_sha256':digest(source),'script_generation_identity':identity,'engine':engine})
    print(f'Script ready: {script_path} ({len(rows)} generated rows)',flush=True)
    return rows
