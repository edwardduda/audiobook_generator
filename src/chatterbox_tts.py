"""Generate a short English narration with local Chatterbox on Apple Silicon."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_FILES = ("ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def ensure_model_files():
    """Use local weights without hub requests; download only missing files."""
    missing = [name for name in MODEL_FILES if not (MODEL_DIR / name).is_file() or (MODEL_DIR / name).stat().st_size == 0]
    if missing:
        from huggingface_hub import hf_hub_download

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for name in missing:
            print(f"Downloading Chatterbox model file: {name}", flush=True)
            hf_hub_download(repo_id="ResembleAI/chatterbox", filename=name, local_dir=MODEL_DIR)
    return MODEL_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Text to speak (a few sentences per call)")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "outputs" / "speech.wav")
    parser.add_argument("--voice", type=Path, help="Reference voice WAV, ideally about 10 seconds")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("Text must not be empty")
    if args.voice and not args.voice.is_file():
        parser.error(f"Voice file does not exist: {args.voice}")

    import torch
    import soundfile as sf
    from chatterbox.tts import ChatterboxTTS

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading Chatterbox on {device}...", flush=True)
    model = ChatterboxTTS.from_local(ensure_model_files(), device=device)
    wav = model.generate(args.text, audio_prompt_path=str(args.voice) if args.voice else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), wav.detach().cpu().numpy().T, model.sr)
    print(f"Saved {args.output.resolve()} ({wav.shape[-1] / model.sr:.1f} seconds)")


if __name__ == "__main__":
    main()
