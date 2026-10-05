"""TTS engine selection and locally cached model weights."""
from __future__ import annotations
import contextlib
import os
from pathlib import Path
from .core import ROOT, PipelineError, file_hash, read_json, save_json

ENGINES = ('dramabox', 'chatterbox')
CHATTERBOX_REPO = 'ResembleAI/chatterbox-turbo'
DRAMABOX_REPO = 'appautomaton/dramabox-tts-3.3b-bf16-mlx'
GEMMA_REPO = 'appautomaton/gemma-3-12b-it-backbone-4bit-mlx'
DRAMABOX_DIR = ROOT/'models/dramabox/mlx-bf16'
GEMMA_DIR = ROOT/'models/gemma_3_12b_it_backbone/mlx-4bit'
DRAMABOX_FILES = {'dramabox-dit-v1.safetensors', 'dramabox-audio-components.safetensors', 'config.json'}
GEMMA_FILES = {'config.json', 'tokenizer.json'}

def engine_name(value=None):
    name = (value or os.getenv('TTS_ENGINE') or 'dramabox').strip().casefold()
    if name not in ENGINES:
        raise PipelineError(f'TTS_ENGINE must be one of: {", ".join(ENGINES)}.')
    return name

def default_model_dir(engine):
    return DRAMABOX_DIR if engine_name(engine) == 'dramabox' else ROOT/'models/turbo'

def gemma_dir():
    return Path(os.getenv('DRAMABOX_GEMMA_DIR') or GEMMA_DIR)

@contextlib.contextmanager
def hub_offline():
    """Load from local files only; never contact Hugging Face during model load."""
    previous = os.environ.get('HF_HUB_OFFLINE')
    os.environ['HF_HUB_OFFLINE'] = '1'
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('HF_HUB_OFFLINE', None)
        else:
            os.environ['HF_HUB_OFFLINE'] = previous

def _local_files(model_dir, marker):
    return [p for p in model_dir.rglob('*') if p.is_file() and p != marker
            and '.cache' not in p.relative_to(model_dir).parts]

def ensure_snapshot(repo, model_dir, required, revision=None):
    """Download a model repo once; later calls verify file sizes without network or re-hashing."""
    model_dir = Path(model_dir)
    marker = model_dir/'model_manifest.json'
    requested = revision or os.getenv('TTS_MODEL_REVISION', 'main')
    if marker.exists():
        manifest = read_json(marker)
        files = manifest.get('files', {})
        if (manifest.get('repo') == repo and manifest.get('requested_revision') == requested and files
                and all((model_dir/name).is_file() and (model_dir/name).stat().st_size == info.get('size')
                        for name, info in files.items())):
            return manifest
    from huggingface_hub import HfApi, snapshot_download
    print(f'Downloading {repo} into {model_dir} (first use only)', flush=True)
    resolved = HfApi().model_info(repo, revision=requested).sha
    snapshot_download(repo, revision=resolved, local_dir=model_dir)
    files = {str(p.relative_to(model_dir)): {'size':p.stat().st_size, 'sha256':file_hash(p)}
             for p in _local_files(model_dir, marker)}
    missing = sorted(set(required) - set(files))
    if missing:
        raise PipelineError(f'{repo} download is incomplete; missing {", ".join(missing)}.')
    manifest = {'repo':repo, 'revision':resolved, 'requested_revision':requested, 'files':files}
    save_json(marker, manifest)
    return manifest

def prepare_dramabox(model_dir=None, text_encoder_dir=None):
    model = ensure_snapshot(DRAMABOX_REPO, Path(model_dir or DRAMABOX_DIR), DRAMABOX_FILES, revision='main')
    encoder = ensure_snapshot(GEMMA_REPO, Path(text_encoder_dir or gemma_dir()), GEMMA_FILES, revision='main')
    return {'repo':model['repo'], 'revision':model['revision'],
            'text_encoder':{'repo':encoder['repo'], 'revision':encoder['revision']}}

def prepare_chatterbox(model_dir):
    from huggingface_hub import HfApi, snapshot_download
    model_dir = Path(model_dir)
    marker = model_dir/'model_manifest.json'
    requested_revision = os.getenv('TTS_MODEL_REVISION','main')
    if marker.exists():
        manifest = read_json(marker)
        if manifest.get('repo') == CHATTERBOX_REPO and manifest.get('requested_revision','main') == requested_revision and all((model_dir/name).is_file() and file_hash(model_dir/name) == sha for name,sha in manifest.get('files',{}).items()) and manifest.get('files'):
            return manifest
    revision = HfApi().model_info(CHATTERBOX_REPO,revision=requested_revision).sha
    snapshot_download(CHATTERBOX_REPO,revision=revision,local_dir=model_dir,
                      allow_patterns=['ve.safetensors','t3_turbo_v1.safetensors','s3gen_meanflow.safetensors',
                                      'conds.pt','tokenizer*.json','added_tokens.json','special_tokens_map.json','vocab.json','merges.txt'])
    files = {str(p.relative_to(model_dir)):file_hash(p) for p in model_dir.iterdir()
             if p.is_file() and p.name != marker.name and p.suffix in {'.safetensors','.json','.txt','.pt','.model'}}
    required = {'ve.safetensors','t3_turbo_v1.safetensors','s3gen_meanflow.safetensors','tokenizer_config.json'}
    tokenizer_present = 'tokenizer.json' in files or {'vocab.json','merges.txt'} <= set(files)
    if not required <= set(files) or not tokenizer_present:
        raise PipelineError('Turbo download is incomplete.')
    manifest = {'repo':CHATTERBOX_REPO,'revision':revision,'requested_revision':requested_revision,'files':files}
    save_json(marker,manifest)
    return manifest

def prepare_engine(engine, model_dir=None):
    engine = engine_name(engine)
    model_dir = Path(model_dir or default_model_dir(engine))
    return prepare_dramabox(model_dir) if engine == 'dramabox' else prepare_chatterbox(model_dir)
