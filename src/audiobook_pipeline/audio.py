"""Chatterbox Turbo / DramaBox synthesis and bounded-memory, manifest-validated assembly."""
from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from . import dramabox
from .core import PipelineError, book_characters, digest, file_hash, load_script, parse_gender, read_json, save_json
from .engines import default_model_dir, engine_name, gemma_dir, prepare_chatterbox, prepare_dramabox
from .voices import load_library, resolve_reference

prepare_model = prepare_chatterbox

def select_device(requested):
    import torch
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available() or requested == 'mps' and not torch.backends.mps.is_available():
        raise PipelineError(f'Requested device {requested} is unavailable; try --device cpu.')
    return requested

def valid_wav(path):
    try:
        with sf.SoundFile(path) as f:
            if f.frames <= 0 or f.channels != 1 or f.samplerate <= 0:
                return False
            nonzero = False
            for block in f.blocks(blocksize=65536):
                if not np.isfinite(block).all():
                    return False
                nonzero |= bool(np.any(np.abs(block)>0.00001))
            return nonzero
    except (RuntimeError,OSError):
        return False

def segment_fingerprint(row,ref_hash,model_identity,settings,prompt=None):
    return digest({'text':prompt or row['dialogue'],'reference':ref_hash,'model':model_identity,'settings':settings,
                   'seed_position':row['position']})

def synthesize(book,library,device='auto',model_dir=None,model=None,model_identity=None,engine=None):
    book = Path(book)
    engine = engine_name(engine)
    rows = load_script(book)
    casting = read_json(book/'casting.json')
    manifest = load_library(library)
    references = {}
    for row in rows:
        sid = row['speaker_id']
        if sid not in casting or casting[sid] not in manifest['voices']:
            raise PipelineError(f'Missing reference voice for {sid}; run cast first.')
        if sid not in references:
            voice = manifest['voices'][casting[sid]]
            references[sid] = (resolve_reference(library,voice),voice['sha256'])
    review = book/'review.json'
    script_engine = read_json(review).get('engine') if review.is_file() else None
    if script_engine and script_engine != engine:
        print(f'Warning: script was annotated for {script_engine} but is being voiced with {engine}; '
              'regenerate the script in a new --book directory for engine-specific directions.',flush=True)
    model_dir = Path(model_dir or default_model_dir(engine))
    if engine == 'dramabox':
        if model is None:
            model_identity = prepare_dramabox(model_dir)
        settings = dramabox.settings()
        characters = book_characters(book)
    else:
        if model is None:
            device = select_device(device)
            model_identity = prepare_chatterbox(model_dir)
        settings = {'temperature':float(os.getenv('TTS_TEMPERATURE','0.8')),'device':device,
                    'engine':'chatterbox-tts-0.1.7','norm_loudness':True,'seed':int(os.getenv('TTS_SEED','0'))}
    index_path = book/'segments/manifest.json'
    index = read_json(index_path) if index_path.exists() else {'version':1,'segments':{}}
    desired = {}
    for row in rows:
        key = str(row['position'])
        sid = row['speaker_id']
        reference,ref_hash = references[sid]
        prompt = None
        if engine == 'dramabox':
            voice_gender = parse_gender(manifest['voices'][casting[sid]].get('gender'))
            prompt = dramabox.build_prompt(row, characters.get(sid), voice_gender)
        fingerprint = segment_fingerprint(row,ref_hash,model_identity,settings,prompt)
        path = book/'segments'/f"{row['position']:08d}.wav"
        cached = index['segments'].get(key,{})
        if cached.get('fingerprint') == fingerprint and path.is_file() and cached.get('sha256') == file_hash(path) and valid_wav(path):
            cached = {**cached, 'speaker_id':sid, 'voice_id':casting[sid]}
            desired[key] = cached
            continue
        seed = settings['seed']+row['position']
        if engine == 'dramabox':
            if model is None:
                print('Loading DramaBox (MLX) from local weights',flush=True)
                model = dramabox.load_model(model_dir, gemma_dir())
            print(f"Generating segment {row['position']}/{len(rows)} ({sid})",flush=True)
            data, sample_rate = dramabox.generate(model, prompt, reference, seed, settings,
                                                  dramabox.spoken_text(row['dialogue']), row['direction'])
        else:
            if model is None:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                print(f'Loading Chatterbox Turbo on {device}',flush=True)
                model = ChatterboxTurboTTS.from_local(model_dir,device=device)
            import torch
            torch.manual_seed(seed)
            print(f"Generating segment {row['position']}/{len(rows)} ({sid})",flush=True)
            wav = model.generate(row['dialogue'],audio_prompt_path=str(reference),temperature=settings['temperature'],norm_loudness=True)
            data, sample_rate = wav.detach().cpu().numpy().reshape(-1), model.sr
        if not len(data) or not np.isfinite(data).all() or not np.any(np.abs(data)>0.00001):
            raise PipelineError(f"Invalid/silent generated audio at position {row['position']}")
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_suffix('.tmp.wav')
        sf.write(temp,np.clip(data,-1,1),sample_rate,subtype='PCM_16')
        temp.replace(path)
        cached = {'fingerprint':fingerprint,'sha256':file_hash(path),'dialogue_sha256':digest(row['dialogue']),
                  'speaker_id':sid,'voice_id':casting[sid],'reference_sha256':ref_hash,
                  'sample_rate':sample_rate,'frames':len(data),'model':model_identity,'settings':settings,
                  'engine':engine, **({'prompt':prompt} if prompt else {})}
        index['segments'][key] = cached
        desired[key] = cached
        save_json(index_path,index)
    index['segments'] = desired
    save_json(index_path,index)
    print(f'Audio segments ready: {book / "segments"}',flush=True)
    return index

def stitch(book,library):
    book = Path(book).resolve()
    rows = load_script(book)
    index = read_json(book/'segments/manifest.json')['segments']
    casting = read_json(book/'casting.json')
    voices = load_library(library)['voices']
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise PipelineError('FFmpeg is required for WAV/RF64 and MP3 assembly.')
    clips = []
    sample_rate = None
    refs = {}
    for row in rows:
        position = str(row['position'])
        path = book/'segments'/f"{row['position']:08d}.wav"
        entry = index.get(position,{})
        voice_id = casting.get(row['speaker_id'])
        if voice_id not in voices:
            raise PipelineError(f'Missing cast voice for {row["speaker_id"]}.')
        if voice_id not in refs:
            resolve_reference(library,voices[voice_id])
            refs[voice_id] = voices[voice_id]['sha256']
        if entry.get('dialogue_sha256') != digest(row['dialogue']) or entry.get('speaker_id') != row['speaker_id'] or entry.get('voice_id') != voice_id or entry.get('reference_sha256') != refs[voice_id] or not path.is_file() or entry.get('sha256') != file_hash(path) or not valid_wav(path):
            raise PipelineError(f'Segment {position} is missing, stale, or corrupt; run synthesize before stitching.')
        info = sf.info(path)
        sample_rate = sample_rate or info.samplerate
        if info.samplerate != sample_rate:
            raise PipelineError('Segment sample rates differ; regenerate with a consistent model.')
        clips.append((path,row['pause_after_ms'],info.frames))
    identity = digest({'clips':[(file_hash(p),pause) for p,pause,_ in clips],'sample_rate':sample_rate,'mp3_bitrate':'128k'})
    output_manifest = book/'audio_manifest.json'
    if output_manifest.exists():
        previous = read_json(output_manifest)
        if previous.get('identity') == identity and all((book/name).is_file() and file_hash(book/name) == sha for name,sha in previous.get('outputs',{}).items()) and len(previous.get('outputs',{})) == 2:
            print(f'Final audio already current: {book}',flush=True)
            return previous
    # Feed PCM in bounded blocks; FFmpeg writes a WAV with RF64 enabled when needed.
    wav_tmp = book/'audiobook.tmp.wav'
    mp3_tmp = book/'audiobook.tmp.mp3'
    log_path = book/'ffmpeg.log'
    try:
        with log_path.open('w') as log:
            process = subprocess.Popen([ffmpeg,'-y','-v','error','-f','s16le','-ar',str(sample_rate),'-ac','1','-i','pipe:0',
                                        '-c:a','pcm_s16le','-rf64','auto',str(wav_tmp)],stdin=subprocess.PIPE,stderr=log)
            try:
                for path,pause,_ in clips:
                    with sf.SoundFile(path) as f:
                        for block in f.blocks(blocksize=65536,dtype='int16'):
                            process.stdin.write(block.astype('<i2').tobytes())
                    process.stdin.write(bytes(round(sample_rate*pause/1000)*2))
                process.stdin.close()
                if process.wait() != 0:
                    raise PipelineError(f'FFmpeg WAV assembly failed; see {log_path}')
            except BaseException:
                process.kill()
                process.wait()
                raise
            subprocess.run([ffmpeg,'-y','-v','error','-i',str(wav_tmp),'-c:a','libmp3lame','-b:a','128k',str(mp3_tmp)],stderr=log,check=True)
        wav_tmp.replace(book/'audiobook.wav')
        mp3_tmp.replace(book/'audiobook.mp3')
    finally:
        wav_tmp.unlink(missing_ok=True)
        mp3_tmp.unlink(missing_ok=True)
    result = {'identity':identity,'duration_seconds':sum(frames/sample_rate+pause/1000 for _,pause,frames in clips),
              'sample_rate':sample_rate,'segments':len(clips),
              'outputs':{name:file_hash(book/name) for name in ('audiobook.wav','audiobook.mp3')}}
    save_json(output_manifest,result)
    print(f'Created {book / "audiobook.wav"} and audiobook.mp3 ({result["duration_seconds"]:.1f}s)',flush=True)
    return result
