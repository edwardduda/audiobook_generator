import io
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from audiobook_pipeline.core import PipelineError, clean_source, normalize, save_json, read_json, split_text, load_script, file_hash, digest
from audiobook_pipeline.script import validate_annotations, generate_script, Truncated, LLMClient, SCHEMAS, quoted_mask
from audiobook_pipeline.voices import build_library, cast_book, decode_candidate, load_library
from audiobook_pipeline.audio import synthesize, stitch, segment_fingerprint
from audiobook_pipeline.engines import engine_name
from audiobook_pipeline import dramabox

REGISTRY = {'narrator':{'name':'Narrator','aliases':[]}}

@pytest.fixture(autouse=True)
def chatterbox_by_default(monkeypatch):
    monkeypatch.setenv('TTS_ENGINE','chatterbox')

def segment(text,speaker='narrator',**kw):
    return {'text':text,'speaker_id':speaker,'direction':'Neutral.','reaction':'','uncertain':False,**kw}

def result(segments,characters=None):
    return {'characters':characters or [],'segments':segments}

def test_metadata_and_fidelity(tmp_path):
    source = tmp_path/'book.txt'
    source.write_text('# source: test.pdf\n\nCHAPTER 1\n\n--- page 2 ---\n\nA quiet room.')
    assert clean_source(source) == 'CHAPTER 1\n\n\nA quiet room.'
    original = 'Mara said, “Come in.”'
    aligned,_,_ = validate_annotations(original,result([segment('Mara said, "Come in."')]),REGISTRY)
    assert aligned[0]['text'] == original

@pytest.mark.parametrize('texts',[['A.','A.','B.'],['B.','A.'],['C.','B.']])
def test_rejects_duplicates_reorder_rewrites(texts):
    with pytest.raises(PipelineError,match='fidelity'):
        validate_annotations('A. B.',result([segment(t) for t in texts]),REGISTRY)

def test_recovers_skipped_and_glued_attribution():
    characters = [{'speaker_id':'tam','name':'Tam','aliases':[]},{'speaker_id':'rand','name':'Rand','aliases':[]}]
    source = 'Tam frowned over Bela\'s back at him. "Are you all right, lad?"\n\n"A rider," Rand said breathlessly, pulling himself upright. "A stranger, following us."'
    aligned,_,warnings = validate_annotations(source,result([
        segment('"Are you all right, lad?"','tam'),
        segment('"A rider," Rand said breathlessly, pulling himself upright.','rand'),
        segment('"A stranger, following us."','rand'),
    ],characters),REGISTRY)
    assert normalize(''.join(r['text'] for r in aligned)) == normalize(source)
    assert aligned[0]['speaker_id'] == 'narrator'
    assert any(r['speaker_id']=='rand' and r['text'].strip().startswith('"A rider') for r in aligned)
    assert any(r['speaker_id']=='narrator' and 'Rand said' in r['text'] for r in aligned)
    assert warnings

def test_attribution_and_aliases():
    characters = [{'speaker_id':'mara','name':'Mara','aliases':['Captain Mara']}]
    aligned,_,warnings = validate_annotations('“Hello,” Mara said.',result([segment('“Hello,” Mara said.','mara')],characters),REGISTRY)
    assert [r['speaker_id'] for r in aligned] == ['mara','narrator']
    assert warnings
    aligned,registry,_ = validate_annotations('“Hello,” Mara said.',result([segment('“Hello,”','mara'),segment('Mara said.')],characters),REGISTRY)
    assert [r['speaker_id'] for r in aligned] == ['mara','narrator']
    assert registry['mara']['gender'] == 'unknown'
    aliases = [{'speaker_id':'captain_mara','name':'Captain Mara','aliases':[]}]
    aligned,registry,_ = validate_annotations('“Hello.”',result([segment('“Hello.”','captain_mara')],aliases),registry)
    assert aligned[0]['speaker_id'] == 'mara'
    assert 'captain_mara' not in registry

def test_uncertain_and_reactions():
    aligned,_,warnings = validate_annotations('“Hello.”',result([segment('“Hello.”','unknown',uncertain=True)]),REGISTRY)
    assert aligned[0]['speaker_id'] == 'narrator' and warnings
    for reaction in ['angry','laugh']:
        with pytest.raises(PipelineError):
            validate_annotations('Hello.',result([segment('Hello.',reaction=reaction)]),REGISTRY)
    aligned,_,_ = validate_annotations('He chuckled. Hello.',result([segment('He chuckled.'),segment('Hello.',reaction='chuckle')]),REGISTRY)
    assert aligned[1]['reaction'] == 'chuckle'

class FakeClient:
    identity = {'model':'fake'}
    def __init__(self, failures=None, sheet=None):
        self.calls = 0
        self.sheet_calls = 0
        self.sheet = list(sheet or [])
        self.failures = list(failures or [])
    def annotate(self,prompt,data,feedback='',schema=None):
        if schema and schema[0] == 'character_sheet':
            self.sheet_calls += 1
            return {'characters':self.sheet}
        self.calls += 1
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure,Exception):
                raise failure
            return failure
        return result([segment(data['source'])])

def test_annotates_whole_source_by_default(tmp_path):
    source = tmp_path/'source.txt'
    source.write_text('A quiet room.\n\nThe sun rose.')
    seen = []
    class Capture(FakeClient):
        def annotate(self,prompt,data,feedback='',schema=None):
            if schema[0] != 'character_sheet':
                seen.append(data)
            return super().annotate(prompt,data,feedback,schema)
    generate_script(source,tmp_path/'book',Capture())
    assert len(seen) == 1
    assert seen[0]['source'] == clean_source(source)
    assert seen[0]['prior_segments'] == []

def test_passes_prior_segments_between_chunks(tmp_path):
    source = tmp_path/'source.txt'
    source.write_text(('A quiet room. ' * 25) + '\n\n' + ('The sun rose. ' * 25))
    seen = []
    class Capture(FakeClient):
        def annotate(self,prompt,data,feedback='',schema=None):
            if schema[0] != 'character_sheet':
                seen.append(data)
            return super().annotate(prompt,data,feedback,schema)
    generate_script(source,tmp_path/'book',Capture(),chunk_chars=350)
    assert len(seen) >= 2
    assert seen[1]['prior_segments']
    assert seen[1]['prior_segments'][0]['speaker_id'] == 'narrator'

def test_resume_retries_and_edit_preservation(tmp_path):
    source = tmp_path/'source.txt'
    source.write_text('A quiet room.\n\nThe sun rose.')
    book = tmp_path/'book'
    client = FakeClient([PipelineError('Invalid JSON'),result([segment('Wrong text.')])])
    rows = generate_script(source,book,client)
    assert client.calls == 3
    assert len(list((book/'diagnostics').glob('*.json'))) == 2
    rows[0]['direction'] = 'User edit'
    (book/'script.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    generate_script(source,book,client)
    assert client.calls == 3
    assert load_script(book)[0]['direction'] == 'User edit'
    source.write_text('Changed source.')
    with pytest.raises(PipelineError,match='changed'):
        generate_script(source,book,client)

def test_truncation_and_long_text(tmp_path):
    source = tmp_path/'source.txt'
    source.write_text('The sun rose over the distant green hills. ' * 25)
    client = FakeClient([Truncated('length')])
    rows = generate_script(source,tmp_path/'book',client,chunk_chars=600,segment_chars=100)
    assert all(len(r['dialogue']) <= 100 for r in rows)
    assert normalize(''.join(r['dialogue'] for r in rows)) == normalize(source.read_text())
    assert [r['position'] for r in rows] == list(range(1,len(rows)+1))

def test_interrupted_chunks_resume(tmp_path):
    source = tmp_path/'source.txt'
    source.write_text(('A quiet room. ' * 25) + '\n\n' + ('The sun rose. ' * 25))
    class Interrupt(FakeClient):
        def annotate(self,prompt,data,feedback='',schema=None):
            if self.calls == 1:
                raise KeyboardInterrupt()
            return super().annotate(prompt,data,feedback,schema)
    client = Interrupt()
    with pytest.raises(KeyboardInterrupt):
        generate_script(source,tmp_path/'book',client,chunk_chars=350)
    checkpoint = json.loads((tmp_path/'book/checkpoints/script.json').read_text())
    assert checkpoint['offset'] > 0
    rows = generate_script(source,tmp_path/'book',FakeClient(),chunk_chars=350)
    assert normalize(''.join(r['dialogue'] for r in rows)) == normalize(source.read_text())

def audio_row(speaker,duration=8,silent=False):
    buffer = io.BytesIO()
    samples = np.zeros(round(duration*24000)) if silent else np.sin(np.arange(round(duration*24000)) * .04)*.1
    sf.write(buffer,samples,24000,format='WAV')
    return {'speaker_id':str(speaker),'id':f'{speaker}_1','text_original':'Original reference.','audio':{'bytes':buffer.getvalue()}}

def make_library(tmp_path,count=3):
    library = tmp_path/'library'
    build_library(library,count=count,rows=[audio_row(i) for i in range(1,count+1)],revision='fixture')
    return library

def write_script(book,rows):
    book.mkdir(exist_ok=True)
    (book/'script.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

def row(position,speaker='narrator',text='Hello.'):
    return {'position':position,'speaker_id':speaker,'dialogue':text,'direction':'unspoken','pause_after_ms':150}

def test_library_selection_and_casting(tmp_path):
    assert decode_candidate(audio_row(1,4)) is None
    assert decode_candidate(audio_row(1,8,True)) is None
    library = tmp_path/'library'
    manifest = build_library(library,count=2,rows=[audio_row(1,14),audio_row(1,10),audio_row(2,8)],revision='fixture')
    assert manifest['voices']['libritts_r_1']['duration'] == 10
    book = tmp_path/'book'
    write_script(book,[row(1),row(2,'mara')])
    cast = cast_book(book,library)
    save_json(book/'casting.json',{'narrator':cast['mara'],'mara':cast['narrator']})
    assert cast_book(book,library)['mara'] == cast['narrator']
    write_script(book,[row(1),row(2,'mara'),row(3,'ben')])
    with pytest.raises(PipelineError,match='additional'):
        cast_book(book,library)

def test_cast_matches_character_gender(tmp_path):
    library = make_library(tmp_path, count=3)
    voices = read_json(library/'manifest.json')['voices']
    for voice_id, gender in zip(sorted(voices), ('female','male','male')):
        voices[voice_id]['gender'] = gender
    save_json(library/'manifest.json', {**read_json(library/'manifest.json'), 'voices':voices})
    book = tmp_path/'book'
    write_script(book,[row(1),row(2,'mara'),row(3,'ben')])
    save_json(book/'characters.json',{
        'narrator':{'name':'Narrator','aliases':[],'gender':'unknown'},
        'mara':{'name':'Mara','aliases':[],'gender':'female'},
        'ben':{'name':'Ben','aliases':[],'gender':'male'},
    })
    cast = cast_book(book,library)
    labeled = load_library(library)['voices']
    assert labeled[cast['mara']]['gender'] == 'female'
    assert labeled[cast['ben']]['gender'] == 'male'

def test_character_gender_from_source_cues():
    characters = [{'speaker_id':'mara','name':'Mara','aliases':[],'gender':'female'}]
    _,registry,_ = validate_annotations('“Hello,” she said.',result([segment('“Hello,”','mara'),segment('she said.')],characters),REGISTRY)
    assert registry['mara']['gender'] == 'female'
    with pytest.raises(PipelineError, match='male, female, or unknown'):
        validate_annotations('“Hi.”',result([segment('“Hi.”','mara')],[{'speaker_id':'mara','name':'Mara','aliases':[],'gender':'child'}]),REGISTRY)

class FakeModel:
    sr = 24000
    def __init__(self):
        self.calls = []
    def generate(self,text,**kwargs):
        import torch
        self.calls.append((text,kwargs))
        return torch.tensor(np.sin(np.arange(2400)*.04)*.1).reshape(1,-1)

def test_synthesis_cache_and_stitch(tmp_path):
    library = make_library(tmp_path)
    book = tmp_path/'book'
    rows = [row(1),row(2,'mara','Welcome.')]
    write_script(book,rows)
    cast_book(book,library)
    model = FakeModel()
    synthesize(book,library,device='cpu',model=model,model_identity='fixture')
    assert len(model.calls) == 2
    assert model.calls[0][1]['audio_prompt_path'] != model.calls[1][1]['audio_prompt_path']
    assert 'unspoken' not in str(model.calls)
    synthesize(book,library,device='cpu',model=model,model_identity='fixture')
    assert len(model.calls) == 2
    rows[1]['dialogue'] = 'Goodbye.'
    write_script(book,rows)
    with pytest.raises(PipelineError,match='stale'):
        stitch(book,library)
    synthesize(book,library,device='cpu',model=model,model_identity='fixture')
    assert len(model.calls) == 3
    output = stitch(book,library)
    assert output['duration_seconds'] == pytest.approx(.5)
    assert sf.info(book/'audiobook.wav').duration == pytest.approx(.5)
    assert (book/'audiobook.mp3').stat().st_size > 0
    assert stitch(book,library) == output
    (book/'segments/00000001.wav').unlink()
    with pytest.raises(PipelineError,match='missing'):
        stitch(book,library)

def test_numeric_order_and_validation(tmp_path):
    book = tmp_path/'book'
    write_script(book,[row(2),row(1)])
    with pytest.raises(PipelineError,match='row 1'):
        load_script(book)
    assert ''.join(split_text('word '*200,100)).replace(' ','') == 'word'*200
    a = segment_fingerprint(row(1),'ref','model',{})
    assert a != segment_fingerprint(row(1),'changed','model',{})
    assert a == segment_fingerprint({**row(1),'direction':'changed','pause_after_ms':900},'ref','model',{})

def test_http_modes_and_invalid_json(monkeypatch):
    import httpx
    for mode in ['json_schema','json_object','text']:
        monkeypatch.setenv('LLM_RESPONSE_FORMAT',mode)
        client = LLMClient()
        client.http.close()
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(result([segment('Hello.')]))}}]})
        client.http = httpx.Client(transport=httpx.MockTransport(handle))
        assert client.annotate('prompt',{'source':'Hello.'})['segments']
        assert ('response_format' in requests[0]) == (mode != 'text')
        client.close()
    client = LLMClient()
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'choices':[{'finish_reason':'length','message':{'content':'{'}}]})))
    with pytest.raises(Truncated):
        client.annotate('prompt',{})
    client.close()
    client = LLMClient()
    client.http.close()
    reasoning_only = {'choices':[{'finish_reason':'stop','message':{'content':'','reasoning_content':'{"segments":[]}'}}]}
    client.http = httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=reasoning_only)))
    assert client.annotate('prompt',{}) == {'segments':[]}
    client.close()

def test_headings_and_paragraph_pauses_are_deterministic(tmp_path):
    source = tmp_path/'source.txt'
    source.write_text('CHAPTER 1\n\nFirst paragraph.\n\nSecond paragraph.\n\nCHAPTER 2\n\nLast paragraph.')
    client = FakeClient()
    rows = generate_script(source,tmp_path/'book',client)
    assert client.calls == 2
    assert [(r['dialogue'],r['pause_after_ms']) for r in rows] == [
        ('CHAPTER 1',900),('First paragraph.',350),('Second paragraph.',900),('CHAPTER 2',900),('Last paragraph.',350)]
    assert normalize(''.join(r['dialogue'] for r in rows)) == normalize(source.read_text())

def test_pronouns_are_not_global_character_aliases():
    characters = [{'speaker_id':'mara','name':'Mara','aliases':['she','Captain Mara']}]
    _,registry,_ = validate_annotations('“Hi.”',result([segment('“Hi.”','mara')],characters),REGISTRY)
    assert registry['mara']['aliases'] == ['Captain Mara']

def test_manual_unsupported_tags_rejected(tmp_path):
    book = tmp_path/'book'
    write_script(book,[row(1,text='[whisper] Hello.')])
    with pytest.raises(PipelineError,match='Unsupported'):
        load_script(book)

def test_reference_edit_invalidates_synthesis(tmp_path):
    library = make_library(tmp_path,1)
    book = tmp_path/'book'
    write_script(book,[row(1)])
    casting = cast_book(book,library)
    model = FakeModel()
    synthesize(book,library,device='cpu',model=model,model_identity='fixture')
    manifest = json.loads((library/'manifest.json').read_text())
    voice = manifest['voices'][casting['narrator']]
    path = library/voice['path']
    data,sr = sf.read(path)
    sf.write(path,data*.7,sr)
    voice['sha256'] = file_hash(path)
    save_json(library/'manifest.json',manifest)
    synthesize(book,library,device='cpu',model=model,model_identity='fixture')
    assert len(model.calls) == 2
    synthesize(book,library,device='cpu',model=model,model_identity='different-model')
    assert len(model.calls) == 3

def test_stitch_preserves_numeric_order_past_nine(tmp_path):
    library = make_library(tmp_path,1)
    book = tmp_path/'book'
    rows = [row(i,text=f'Line {i}.') for i in range(1,13)]
    write_script(book,rows)
    cast_book(book,library)
    synthesize(book,library,device='cpu',model=FakeModel(),model_identity='fixture')
    index = json.loads((book/'segments/manifest.json').read_text())
    expected = []
    for item in rows:
        position = item['position']
        path = book/'segments'/f'{position:08d}.wav'
        data = np.full(2400,position*100,dtype=np.int16)
        sf.write(path,data,24000,subtype='PCM_16')
        index['segments'][str(position)]['sha256'] = file_hash(path)
        expected.extend([data,np.zeros(3600,dtype=np.int16)])
    save_json(book/'segments/manifest.json',index)
    stitch(book,library)
    actual,_ = sf.read(book/'audiobook.wav',dtype='int16')
    np.testing.assert_array_equal(actual,np.concatenate(expected))

def test_tiny_truncated_chunk_does_not_repeat_identical_requests(tmp_path):
    source = tmp_path/'source.txt'
    source.write_text('A short passage.')
    client = FakeClient([Truncated('length')])
    with pytest.raises(PipelineError,match='minimum chunk'):
        generate_script(source,tmp_path/'book',client)
    assert client.calls == 1

def test_engine_selection(monkeypatch):
    monkeypatch.delenv('TTS_ENGINE')
    assert engine_name() == 'dramabox'
    monkeypatch.setenv('TTS_ENGINE','Chatterbox')
    assert engine_name() == 'chatterbox'
    assert engine_name('dramabox') == 'dramabox'
    monkeypatch.setenv('TTS_ENGINE','bark')
    with pytest.raises(PipelineError,match='TTS_ENGINE'):
        engine_name()

SHEET = [{'speaker_id':'mara','name':'Mara','aliases':['Captain Mara'],'gender':'female','gender_evidence':'she said',
          'age':'elderly','voice':'a raspy, weary voice','personality':'proud'}]

def delivery_segment(text,speaker='narrator',delivery='The narrator reads softly.'):
    return {'text':text,'speaker_id':speaker,'delivery':delivery,'uncertain':False}

def test_dramabox_script_uses_engine_prompt_schema_and_character_sheet(tmp_path):
    source = tmp_path/'source.txt'
    source.write_text('Mara laughed. “Come in,” she whispered. “Sit,” Ben said.')
    seen = []
    class Capture(FakeClient):
        def annotate(self,prompt,data,feedback='',schema=None):
            seen.append((prompt,data,schema))
            if schema[0] == 'character_sheet':
                return super().annotate(prompt,data,feedback,schema)
            return {'characters':[{'speaker_id':'ben','name':'Ben','aliases':[],'gender':'male'}],'segments':[
                delivery_segment('Mara laughed.'),
                delivery_segment('“Come in,”','mara','She whispers through a laugh.'),
                delivery_segment('she whispered.'),
                delivery_segment('“Sit,”','ben','He speaks gruffly.'),
                delivery_segment('Ben said.')]}
    book = tmp_path/'book'
    rows = generate_script(source,book,Capture(sheet=SHEET),engine='dramabox')
    sheet_call, annotate_call = seen
    assert sheet_call[2][0] == 'character_sheet' and 'gender_evidence' in json.dumps(sheet_call[2][1])
    assert annotate_call[0] == (Path(__file__).parents[1]/'prompts/dramabox_system.txt').read_text()
    assert annotate_call[2][1] == SCHEMAS['dramabox']
    assert 'reaction' not in json.dumps(SCHEMAS['dramabox'])
    assert annotate_call[1]['characters']['mara']['gender'] == 'female'
    assert rows[1]['speaker_id'] == 'mara' and rows[1]['direction'] == 'She whispers through a laugh.'
    assert read_json(book/'character_sheet.json')['mara']['voice'] == 'a raspy, weary voice'
    review = read_json(book/'review.json')
    assert review['engine'] == 'dramabox'
    assert any('character_sheet' in w['reason'] and w['text'] == 'Ben' for w in review['warnings'])
    with pytest.raises(PipelineError,match='engine'):
        generate_script(source,book,Capture(),engine='chatterbox')

def test_dramabox_requires_delivery():
    with pytest.raises(PipelineError,match='segment fields'):
        validate_annotations('Hello.',result([segment('Hello.')]),REGISTRY,engine='dramabox')

def test_edited_character_sheet_is_reused_and_drives_casting(tmp_path):
    source = tmp_path/'source.txt'
    source.write_text('“Hello,” Mara said.')
    book = tmp_path/'book'
    book.mkdir()
    save_json(book/'character_sheet.json',{'mara':{'name':'Mara','aliases':[],'gender':'male'}})
    client = FakeClient(sheet=SHEET)
    generate_script(source,book,client)
    assert client.sheet_calls == 0
    library = make_library(tmp_path,count=2)
    voices = read_json(library/'manifest.json')['voices']
    for voice_id, gender in zip(sorted(voices), ('female','male')):
        voices[voice_id]['gender'] = gender
    save_json(library/'manifest.json',{**read_json(library/'manifest.json'),'voices':voices})
    write_script(book,[row(1),row(2,'mara')])
    save_json(book/'characters.json',{'mara':{'name':'Mara','aliases':[],'gender':'female'}})
    cast = cast_book(book,library)
    assert load_library(library)['voices'][cast['mara']]['gender'] == 'male'

def test_dramabox_prompt_building():
    character = {'gender':'female','age':'elderly','voice':'raspy, weary'}
    prompt = dramabox.build_prompt({'speaker_id':'mara','dialogue':'“He said “no” to me,”','direction':'She hisses with contempt.'},character)
    assert prompt == 'An elderly woman with a raspy, weary voice hisses with contempt, "He said \'no\' to me,"'
    assert prompt.count('"') == 2
    narrator = dramabox.build_prompt({'speaker_id':'narrator','dialogue':'Rain fell.','direction':'Neutral narration.'},{},'male')
    assert narrator == 'A male narrator with a clear, warm, engaging voice reads in a steady, engaging tone, "Rain fell."'
    unknown = dramabox.build_prompt({'speaker_id':'ben','dialogue':'[laugh] Hi.','direction':''},{'age':'child'})
    assert unknown == 'A child speaks naturally, "Hi."'
    leaky = dramabox.build_prompt({'speaker_id':'narrator','dialogue':'Rain fell.','direction':'The narrator describes the setting vividly.'},{})
    assert leaky == 'A narrator with a clear, warm, engaging voice reads vividly, "Rain fell."'

def test_dramabox_delivery_keeps_only_vocal_performance():
    cases = {
        'The narrator introduces the scene with a somber tone.': 'The narrator reads with a somber tone.',
        'The narrator describes the setting and action vividly.': 'The narrator reads vividly.',
        'The narrator reads urgently to match her outburst.': 'The narrator reads urgently.',
        'The narrator reads with tension as she composes herself.': 'The narrator reads with tension.',
        'The narrator reads quietly, matching his murmur.': 'The narrator reads quietly.',
        'The narrator describes the room.': '',
        'Neutral narration.': 'Neutral narration.',
    }
    for delivery, expected in cases.items():
        assert dramabox.clean_delivery(delivery) == expected
    assert dramabox.clean_delivery('She screams in anger, emphasizing the years.','mara') == 'She screams in anger.'
    assert dramabox.clean_delivery('She commands him with a mix of affection and authority.','mara') == 'She speaks with a mix of affection and authority.'
    assert dramabox.clean_delivery('She laughs bitterly, her voice sharp with contempt.','mara') == 'She laughs bitterly, her voice sharp with contempt.'
    assert dramabox.clean_delivery('She asks with a startled, confused tone.','mara') == 'She speaks with a startled, confused tone.'
    assert dramabox.clean_delivery('He speaks in a flat, commanding tone.','ben') == 'He speaks in a flat, commanding tone.'
    assert dramabox.clean_delivery('The narrator reads with a descriptive, setting-the-scene tone.') == 'The narrator reads with a descriptive, setting-the-scene tone.'

def test_unclosed_quote_does_not_invert_later_text():
    source = '"What did it cost "Just a squirrel," says Gale. "Even wished me luck."\n\n"Well," I say. Prim left.'
    mask = quoted_mask(source)
    quoted = lambda s: all(mask[i] for i in range(source.index(s), source.index(s) + len(s)))
    unquoted = lambda s: not any(mask[i] for i in range(source.index(s), source.index(s) + len(s)))
    assert quoted('"Just a squirrel,"') and quoted('"Even wished me luck."') and quoted('"Well,"')
    assert unquoted('says Gale.') and unquoted('I say. Prim left.')
    paragraph = quoted_mask('"Never closed.\n\nNarration here.')
    assert not any(paragraph[-len('Narration here.'):])

def test_dramabox_script_strips_quotes_and_content_directions(tmp_path):
    source = tmp_path/'source.txt'
    source.write_text('Rain fell. “You kept this from me?” she shouted.')
    class Capture(FakeClient):
        def annotate(self,prompt,data,feedback='',schema=None):
            if schema[0] == 'character_sheet':
                return super().annotate(prompt,data,feedback,schema)
            return {'characters':[],'segments':[
                delivery_segment('Rain fell.',delivery='The narrator describes the setting calmly.'),
                delivery_segment('“You kept this from me?”','mara','She shouts with fury, commanding him.'),
                delivery_segment('she shouted.')]}
    book = tmp_path/'book'
    rows = generate_script(source,book,Capture(sheet=SHEET),engine='dramabox')
    assert [(r['dialogue'],r['direction']) for r in rows] == [
        ('Rain fell.','The narrator reads calmly.'),
        ('You kept this from me?','She shouts with fury.'),
        ('she shouted.','The narrator reads softly.')]
    assert sum('Delivery trimmed' in w['reason'] for w in read_json(book/'review.json')['warnings']) == 2

def test_dramabox_duration_and_trim():
    assert dramabox.estimate_duration('Hi.') == pytest.approx(1.5)
    assert dramabox.estimate_duration('word ' * 400) == 35.0
    assert dramabox.estimate_duration('word ' * 40, speed=2.0) < dramabox.estimate_duration('word ' * 40)
    sr = 48000
    tone = np.sin(np.arange(sr)*.05).astype(np.float32)*.3
    padded = np.concatenate([np.zeros(sr),tone,np.zeros(2*sr)]).astype(np.float32)
    trimmed = dramabox.trim_silence(padded,sr)
    assert len(tone) <= len(trimmed) <= len(tone) + round(sr*.2)
    assert dramabox.to_mono(np.stack([tone,tone])).shape == tone.shape

class FakeDramaBox:
    def __init__(self):
        self.calls = []
    def generate(self,prompt,**kwargs):
        from types import SimpleNamespace
        self.calls.append((prompt,kwargs))
        tone = np.sin(np.arange(4800)*.05).astype(np.float32)*.2
        wave = np.stack([np.concatenate([np.zeros(9600,np.float32),tone,np.zeros(9600,np.float32)])]*2)
        return SimpleNamespace(waveform=wave,sample_rate=48000)

def test_dramabox_synthesis_cache_prompt_and_stitch(tmp_path, monkeypatch):
    monkeypatch.setenv('TTS_ENGINE','dramabox')
    library = make_library(tmp_path)
    book = tmp_path/'book'
    rows = [row(1,text='Rain fell.'),row(2,'mara','“Come in.”')]
    rows[1]['direction'] = 'She whispers.'
    write_script(book,rows)
    save_json(book/'character_sheet.json',{'mara':{'name':'Mara','aliases':[],'gender':'female','age':'young','voice':'a bright voice'}})
    cast_book(book,library)
    model = FakeDramaBox()
    synthesize(book,library,model=model,model_identity='fixture')
    assert len(model.calls) == 2
    prompt, kwargs = model.calls[1]
    assert prompt == 'A young woman with a bright voice whispers, "Come in."'
    assert kwargs['voice_ref'] != model.calls[0][1]['voice_ref'] and 1.5 <= kwargs['duration_s'] <= 35
    assert sf.info(book/'segments/00000002.wav').samplerate == 48000
    assert sf.info(book/'segments/00000002.wav').channels == 1
    assert sf.info(book/'segments/00000002.wav').frames < 14400
    synthesize(book,library,model=model,model_identity='fixture')
    assert len(model.calls) == 2
    rows[1]['direction'] = 'She shouts with fury.'
    write_script(book,rows)
    synthesize(book,library,model=model,model_identity='fixture')
    assert len(model.calls) == 3 and 'shouts with fury' in model.calls[-1][0]
    output = stitch(book,library)
    assert output['sample_rate'] == 48000
