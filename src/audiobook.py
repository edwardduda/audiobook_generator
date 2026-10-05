#!/usr/bin/env python3
"""Create a faithful, resumable multi-voice audiobook from extracted UTF-8 text."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
ENGINES = ('dramabox','chatterbox')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command',required=True)
    voices = commands.add_parser('voices',help='Build or audition a reference library')
    voice_commands = voices.add_subparsers(dest='voice_command',required=True)
    build = voice_commands.add_parser('build')
    build.add_argument('--library',type=Path,default=ROOT/'voices/libritts_r')
    build.add_argument('--count',type=int,default=20)
    build.add_argument('--split',default='dev.clean')
    build.add_argument('--revision',default='main')
    build.add_argument('--max-rows',type=int,default=10000)
    listen = voice_commands.add_parser('audition')
    listen.add_argument('voice_id')
    listen.add_argument('--library',type=Path,default=ROOT/'voices/libritts_r')
    listed = voice_commands.add_parser('list')
    listed.add_argument('--library',type=Path,default=ROOT/'voices/libritts_r')
    models = commands.add_parser('models',help='Download TTS weights into models/ for offline use')
    model_commands = models.add_subparsers(dest='model_command',required=True)
    download = model_commands.add_parser('download')
    download.add_argument('--engine',choices=ENGINES,help='Defaults to TTS_ENGINE (dramabox)')
    download.add_argument('--all',action='store_true',help='Download weights for every engine')
    for name in ('script','cast','synthesize','stitch','run'):
        command = commands.add_parser(name)
        command.add_argument('--book',type=Path,required=True,help='Book output directory')
        if name in ('script','synthesize','run'):
            command.add_argument('--engine',choices=ENGINES,help='TTS engine; defaults to TTS_ENGINE (dramabox)')
        if name in ('script','run'):
            command.add_argument('input',type=Path)
            command.add_argument('--chunk-chars',type=int,default=0,help='Source characters per LLM request. 0 sends the whole remaining source so speakers stay in memory.')
            command.add_argument('--segment-chars',type=int,default=300)
            command.add_argument('--prompt',type=Path,help='Annotation prompt; defaults to prompts/<engine>_system.txt')
            command.add_argument('--sheet-prompt',type=Path,default=ROOT/'prompts/character_sheet_system.txt')
        if name != 'script':
            command.add_argument('--library',type=Path,default=ROOT/'voices/libritts_r')
        if name in ('synthesize','run'):
            command.add_argument('--device',choices=['auto','cuda','mps','cpu'],default=os.getenv('TTS_DEVICE','auto'),help='Chatterbox only; DramaBox always runs on MLX')
            command.add_argument('--model-dir',type=Path,help='Defaults to models/dramabox/mlx-bf16 or models/turbo')
        if name == 'run':
            command.add_argument('--voice-count',type=int,default=20)
    return p


def main(argv=None):
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT/'.env',override=False)
        from audiobook_pipeline.core import PipelineError
        from audiobook_pipeline.engines import engine_name, prepare_engine, ENGINES as AVAILABLE
        from audiobook_pipeline.voices import build_library, cast_book, audition, list_voices
        from audiobook_pipeline.script import LLMClient, generate_script
        from audiobook_pipeline.audio import synthesize, stitch
        args = parser().parse_args(argv)
        if args.command == 'voices':
            if args.voice_command == 'build':
                build_library(args.library,args.count,args.split,args.revision,args.max_rows)
            elif args.voice_command == 'list':
                list_voices(args.library)
            else:
                audition(args.library,args.voice_id)
            return 0
        if args.command == 'models':
            for engine in (AVAILABLE if args.all else [engine_name(args.engine)]):
                prepare_engine(engine)
                print(f'{engine} weights ready (local, offline loading)',flush=True)
            return 0
        engine = engine_name(getattr(args,'engine',None))
        if args.command in ('script','run'):
            if args.chunk_chars < 0 or 0 < args.chunk_chars < 256 or args.segment_chars < 32:
                raise PipelineError('--chunk-chars must be 0 (whole source) or >=256, and --segment-chars >=32.')
            client = LLMClient()
            try:
                generate_script(args.input,args.book,client,args.chunk_chars,args.segment_chars,args.prompt,
                                engine,args.sheet_prompt)
            finally:
                client.close()
        if args.command == 'run':
            build_library(args.library,count=args.voice_count)
        if args.command in ('cast','run'):
            cast_book(args.book,args.library)
        if args.command in ('synthesize','run'):
            synthesize(args.book,args.library,args.device,args.model_dir,engine=engine)
        if args.command in ('stitch','run'):
            stitch(args.book,args.library)
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Completed stages and segments can be resumed.',file=sys.stderr)
        return 130
    except Exception as e:
        if os.getenv('AUDIOBOOK_DEBUG') == '1':
            raise
        print(f'Error: {e}',file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
