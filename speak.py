#!/usr/bin/env python3
"""
Orpheus TTS via Ollama — speaks any text you provide.
Usage:
  python speak.py "Your text here"
  python speak.py --file <path>           (read from file)
  python speak.py --file <path> --save <output.wav>  (generate and save)
  python speak.py              (interactive mode)
"""

import sys
import os
import re
import time
import numpy as np
import sounddevice as sd
import requests
import torch
from snac import SNAC
from threading import Thread
from queue import Queue
from scipy.io import wavfile

# ── Config ───────────────────────────────────────────────────────────────────
# Ollama/Orpheus settings
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "legraphista/Orpheus:3b-ft-q8"
VOICE       = "tara"   # tara | leah | jess | leo | dan | mia | zac | zoe
SAMPLE_RATE = 24000
MAX_CHARS_PER_CHUNK = 300  # Split long texts into chunks of this size (reduced to avoid token limit)

# MLX-Audio settings
MLX_MODELS = {
    "pocket": "mlx-community/pocket-tts-4bit",  # Fast, compact model
    "kokoro": "mlx-community/Kokoro-82M-bf16",  # High-quality Kokoro model
}
DEFAULT_MLX_MODEL = "pocket"  # Default MLX model to use

# Kokoro voice options
KOKORO_VOICES = ["af_heart", "af_bella", "af_nicole", "am_fenrir"]
DEFAULT_KOKORO_VOICE = "af_heart"

MLX_SAMPLE_RATE = 24000

# Default backend: "ollama" or "mlx"
DEFAULT_BACKEND = "ollama"
# ─────────────────────────────────────────────────────────────────────────────

snac_model = None
mlx_models = {}  # Cache for loaded MLX models

def load_snac_model():
    global snac_model
    if snac_model is None:
        print("Loading SNAC model...", end=" ", flush=True)
        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        print("ready.\n")

def load_mlx_model(model_name: str = DEFAULT_MLX_MODEL):
    """Load MLX-Audio model."""
    global mlx_models
    if model_name not in mlx_models:
        model_path = MLX_MODELS.get(model_name)
        if not model_path:
            raise ValueError(f"Unknown MLX model: {model_name}. Available: {list(MLX_MODELS.keys())}")
        print(f"Loading MLX-Audio model ({model_path})...", end=" ", flush=True)
        from mlx_audio.tts import load
        mlx_models[model_name] = load(model_path)
        print("ready.\n")
    return mlx_models[model_name]

def generate_audio_mlx(text: str, model_name: str = DEFAULT_MLX_MODEL, voice: str = None) -> np.ndarray | None:
    """Generate audio using MLX-Audio backend."""
    model = load_mlx_model(model_name)

    # Generate speech with voice parameter for Kokoro
    if model_name == "kokoro" and voice:
        audio = model.generate(text, voice=voice)
    else:
        audio = model.generate(text)

    # Collect audio chunks from generator
    audio_chunks = []
    sample_rate = None

    for result in audio:
        audio_chunks.append(np.array(result.audio))
        sample_rate = result.sample_rate

    if not audio_chunks:
        return None

    # Combine all audio chunks
    audio_array = np.concatenate(audio_chunks)

    # Ensure float32 in range [-1, 1]
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)

    return audio_array

def format_prompt(voice: str, text: str) -> str:
    return f"<custom_token_3><|begin_of_text|>{voice}: {text}<|eot_id|><custom_token_4><custom_token_5><custom_token_1>"

def turn_token_into_id(token_str: str, index: int) -> int | None:
    m = re.match(r"<custom_token_(\d+)>", token_str)
    if not m:
        return None
    raw_id = int(m.group(1))
    # Apply Orpheus token ID conversion formula
    token_id = raw_id - 10 - ((index % 7) * 4096)
    return token_id

def tokens_to_audio(token_strings: list[str]) -> np.ndarray | None:
    # Pass index to conversion function for proper offset calculation
    ids = [turn_token_into_id(t, i) for i, t in enumerate(token_strings)]
    ids = [x for x in ids if x is not None]
    ids = ids[:len(ids) - (len(ids) % 7)]
    if not ids:
        return None

    audio_ids = torch.tensor(ids, dtype=torch.int32).reshape(-1, 7)

    codes_0 = audio_ids[:, 0].unsqueeze(0)
    codes_1 = torch.stack((audio_ids[:, 1], audio_ids[:, 4])).t().flatten().unsqueeze(0)
    codes_2 = (
        torch.stack((audio_ids[:, 2], audio_ids[:, 3], audio_ids[:, 5], audio_ids[:, 6]))
        .t().flatten().unsqueeze(0)
    )

    with torch.inference_mode():
        audio_hat = snac_model.decode([codes_0, codes_1, codes_2])

    return audio_hat[0].squeeze().numpy()

def split_text_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If a single sentence is too long, split it further
        if len(sentence) > max_chars:
            # If we have a current chunk, save it
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Split long sentence by commas or spaces
            words = re.split(r'([,;]\s+|\s+)', sentence)
            for word in words:
                if len(current_chunk) + len(word) > max_chars and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    current_chunk += word
        # If adding this sentence exceeds limit, save current chunk
        elif len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += (" " + sentence if current_chunk else sentence)

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def speak_chunk(text: str, voice: str = VOICE, chunk_num: int = None, total_chunks: int = None):
    """Speak a single chunk of text."""
    chunk_info = f" (chunk {chunk_num}/{total_chunks})" if chunk_num else ""
    preview = text[:50] + "..." if len(text) > 50 else text
    print(f"🎤  '{preview}'{chunk_info}  →  generating...", flush=True)

    prompt = format_prompt(voice, text)

    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {
            "temperature": 0.6,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "num_predict": 2048,  # Increased to allow longer chunks
        }
    }, timeout=180)

    resp.raise_for_status()
    raw_text = resp.json().get("response", "")

    token_strings = re.findall(r"<custom_token_\d+>", raw_text)
    print(f"   Found {len(token_strings)} audio tokens.", flush=True)

    if not token_strings:
        print("⚠️  No audio tokens. Raw response preview:")
        print(raw_text[:300])
        return

    waveform = tokens_to_audio(token_strings)
    if waveform is None or len(waveform) == 0:
        print("⚠️  Token decoding produced no audio.")
        return

    duration = len(waveform) / SAMPLE_RATE
    print(f"✅  Playing {duration:.1f}s of audio...")
    sd.play(waveform, samplerate=SAMPLE_RATE)
    time.sleep(duration + 0.1)
    sd.stop()
    print("   Done.\n")

def generate_audio_for_chunk(text: str, voice: str, chunk_num: int, total_chunks: int) -> np.ndarray | None:
    """Generate audio waveform for a chunk without playing it."""
    chunk_info = f" (chunk {chunk_num}/{total_chunks})" if chunk_num else ""
    preview = text[:50] + "..." if len(text) > 50 else text
    print(f"🎤  '{preview}'{chunk_info}  →  generating...", flush=True)

    prompt = format_prompt(voice, text)

    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {
            "temperature": 0.6,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "num_predict": 2048,  # Increased to allow longer chunks
        }
    }, timeout=180)

    resp.raise_for_status()
    raw_text = resp.json().get("response", "")

    token_strings = re.findall(r"<custom_token_\d+>", raw_text)
    print(f"   Found {len(token_strings)} audio tokens.", flush=True)

    if not token_strings:
        print("⚠️  No audio tokens. Raw response preview:")
        print(raw_text[:300])
        return None

    waveform = tokens_to_audio(token_strings)
    if waveform is None or len(waveform) == 0:
        print("⚠️  Token decoding produced no audio.")
        return None

    print(f"   Generated waveform: {len(waveform)} samples ({len(waveform)/SAMPLE_RATE:.2f}s)")

    return waveform

def audio_generator_thread(chunks: list[str], voice: str, audio_queue: Queue):
    """Thread function to generate audio chunks and put them in the queue."""
    for i, chunk in enumerate(chunks, 1):
        waveform = generate_audio_for_chunk(chunk, voice, i, len(chunks))
        audio_queue.put(waveform)
    audio_queue.put(None)  # Signal completion

def speak_mlx(text: str, model_name: str = DEFAULT_MLX_MODEL, voice: str = None, stream: bool = True):
    """Speak text using MLX-Audio backend with optional streaming."""
    preview = text[:50] + "..." if len(text) > 50 else text
    voice_info = f", voice: {voice}" if voice and model_name == "kokoro" else ""
    print(f"🎤  '{preview}'  →  generating with MLX ({model_name}{voice_info})...", flush=True)

    # Use streaming for longer texts
    if stream and len(text) > 100:
        model = load_mlx_model(model_name)

        # Generate speech with voice parameter for Kokoro
        if model_name == "kokoro" and voice:
            audio_gen = model.generate(text, voice=voice)
        else:
            audio_gen = model.generate(text)

        # Stream audio chunks as they're generated
        chunk_num = 0
        for result in audio_gen:
            chunk_num += 1
            chunk_audio = np.array(result.audio)

            # Ensure float32
            if chunk_audio.dtype != np.float32:
                chunk_audio = chunk_audio.astype(np.float32)

            duration = len(chunk_audio) / MLX_SAMPLE_RATE

            if chunk_num == 1:
                print(f"▶️  Streaming audio (chunk {chunk_num})...", flush=True)

            # Play this chunk
            sd.play(chunk_audio, samplerate=MLX_SAMPLE_RATE)
            sd.wait()  # Wait for chunk to finish before playing next

        print(f"✅  Completed streaming {chunk_num} chunks.\n")
    else:
        # For short text, use the standard non-streaming approach
        waveform = generate_audio_mlx(text, model_name, voice)
        if waveform is None or len(waveform) == 0:
            print("⚠️  Audio generation produced no output.")
            return

        duration = len(waveform) / MLX_SAMPLE_RATE
        print(f"✅  Playing {duration:.1f}s of audio...")
        sd.play(waveform, samplerate=MLX_SAMPLE_RATE)
        time.sleep(duration + 0.1)
        sd.stop()
        print("   Done.\n")

def generate_full_audio(text: str, voice: str = VOICE) -> np.ndarray | None:
    """Generate complete audio for text, returning concatenated waveform."""
    load_snac_model()

    # Split into chunks if needed
    if len(text) <= MAX_CHARS_PER_CHUNK:
        chunks = [text]
    else:
        chunks = split_text_into_chunks(text, MAX_CHARS_PER_CHUNK)
        print(f"📄 Text is {len(text)} chars, splitting into {len(chunks)} chunks...\n")

    # Generate all audio chunks
    all_waveforms = []
    for i, chunk in enumerate(chunks, 1):
        waveform = generate_audio_for_chunk(chunk, voice, i, len(chunks))
        if waveform is not None:
            all_waveforms.append(waveform)

    if not all_waveforms:
        print("⚠️  No audio generated.")
        return None

    # Concatenate all waveforms into one continuous audio stream
    if len(all_waveforms) > 1:
        print(f"\n🔗 Concatenating {len(all_waveforms)} audio chunks...")
    combined_waveform = np.concatenate(all_waveforms)

    return combined_waveform

def save_audio_mlx(text: str, output_path: str, model_name: str = DEFAULT_MLX_MODEL, voice: str = None):
    """Generate audio using MLX and save to WAV file."""
    voice_info = f", voice: {voice}" if voice and model_name == "kokoro" else ""
    print(f"🎤  Generating audio with MLX ({model_name}{voice_info})...", flush=True)
    waveform = generate_audio_mlx(text, model_name, voice)

    if waveform is None:
        print("⚠️  No audio generated.")
        return

    total_duration = len(waveform) / MLX_SAMPLE_RATE
    print(f"💾 Saving {total_duration:.1f}s of audio to {output_path}...")

    # Debug: check waveform stats
    print(f"   Waveform shape: {waveform.shape}")
    print(f"   Value range: [{waveform.min():.3f}, {waveform.max():.3f}]")

    # Convert to int16 for WAV format
    audio_int16 = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, MLX_SAMPLE_RATE, audio_int16)

    print(f"✅ Audio saved successfully!\n")

def save_audio(text: str, output_path: str, voice: str = VOICE, backend: str = DEFAULT_BACKEND, mlx_model: str = DEFAULT_MLX_MODEL, kokoro_voice: str = None):
    """Generate audio and save to WAV file."""
    if backend == "mlx":
        save_audio_mlx(text, output_path, mlx_model, kokoro_voice)
        return

    combined_waveform = generate_full_audio(text, voice)

    if combined_waveform is None:
        return

    total_duration = len(combined_waveform) / SAMPLE_RATE
    print(f"💾 Saving {total_duration:.1f}s of audio to {output_path}...")

    # Debug: check waveform stats
    print(f"   Waveform shape: {combined_waveform.shape}")
    print(f"   Value range: [{combined_waveform.min():.3f}, {combined_waveform.max():.3f}]")

    # Convert to int16 for WAV format
    # Ensure proper clipping to avoid overflow
    audio_int16 = np.clip(combined_waveform * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, SAMPLE_RATE, audio_int16)

    print(f"✅ Audio saved successfully!\n")

def speak(text: str, voice: str = VOICE, backend: str = DEFAULT_BACKEND, mlx_model: str = DEFAULT_MLX_MODEL, kokoro_voice: str = None):
    """Speak text, splitting into chunks if necessary."""
    # Route to MLX backend if specified
    if backend == "mlx":
        speak_mlx(text, mlx_model, kokoro_voice)
        return

    load_snac_model()

    # If text is short enough, speak it directly
    if len(text) <= MAX_CHARS_PER_CHUNK:
        speak_chunk(text, voice)
    else:
        # Split into chunks and process them with continuous playback
        chunks = split_text_into_chunks(text, MAX_CHARS_PER_CHUNK)
        print(f"📄 Text is {len(text)} chars, splitting into {len(chunks)} chunks...\n")

        # Create a queue for audio chunks
        audio_queue = Queue(maxsize=2)  # Buffer up to 2 chunks ahead

        # Start generation thread
        generator = Thread(target=audio_generator_thread, args=(chunks, voice, audio_queue))
        generator.start()

        # Play audio chunks as they become available
        chunk_num = 0
        while True:
            waveform = audio_queue.get()
            if waveform is None:  # End signal
                break
            chunk_num += 1
            duration = len(waveform) / SAMPLE_RATE
            print(f"▶️  Playing chunk {chunk_num}/{len(chunks)} ({duration:.1f}s)...")

            # Stop any previous playback to ensure clean state
            sd.stop()

            # Play the chunk
            sd.play(waveform, samplerate=SAMPLE_RATE)

            # Wait for playback to complete using the actual duration
            # Add a small buffer to ensure complete playback
            time.sleep(duration + 0.1)

            # Ensure stream has stopped
            sd.stop()

            print("   Done.\n")

        generator.join()
        print("✅ All chunks completed!")

def show_help():
    print("""
Orpheus TTS — Text-to-speech with multiple backends

Usage:
  python speak.py "Your text here"                  Speak the provided text
  python speak.py --file <path>                     Read and speak from a text file
  python speak.py -f <path>                         Short form of --file
  python speak.py --file <path> --save <output.wav> Generate and save to WAV file
  python speak.py "text" --save <output.wav>        Generate from text and save
  python speak.py --backend mlx "Hello world"       Use MLX-Audio backend
  python speak.py                                   Enter interactive mode
  python speak.py --help                            Show this help message

Options:
  --file, -f <path>         Read text from the specified file
  --save, -s <path>         Save audio to WAV file instead of playing
  --backend, -b <name>      Backend to use: "ollama" or "mlx" (default: {BACKEND})
  --mlx-model, -m <name>    MLX model: "pocket" or "kokoro" (default: {DEFAULT_MLX_MODEL})
  --kokoro-voice, -v <name> Kokoro voice: "af_heart", "af_bella", "af_nicole", "am_fenrir" (default: {DEFAULT_KOKORO_VOICE})
  --help, -h                Show this help message and exit

Configuration (Ollama backend):
  Voice: {VOICE}
  Model: {MODEL}
  Sample Rate: {SAMPLE_RATE} Hz
  Ollama URL: {OLLAMA_URL}
  Max Chunk Size: {MAX_CHARS} characters

Configuration (MLX backend):
  Default Model: {DEFAULT_MLX_MODEL}
  Available Models:
    - pocket: {POCKET_MODEL} (fast, compact)
    - kokoro: {KOKORO_MODEL} (high-quality)
  Kokoro Voices: {KOKORO_VOICES_STR}
  Default Kokoro Voice: {DEFAULT_KOKORO_VOICE}
  Sample Rate: {MLX_SAMPLE_RATE} Hz

Available voices (Ollama only):
  tara, leah, jess, leo, dan, mia, zac, zoe

Notes:
  - Ollama backend: Long texts are split into chunks, streamed playback
  - MLX backend: Fast local inference using Apple Silicon, streaming playback
  - MLX streaming: Audio starts playing immediately for texts > 100 chars
  - Kokoro voices: Use --kokoro-voice to select (only works with kokoro model)
  - Saved audio is in WAV format, 16-bit PCM
  - To change settings, edit the config variables in the script
""".format(VOICE=VOICE, MODEL=MODEL, SAMPLE_RATE=SAMPLE_RATE, OLLAMA_URL=OLLAMA_URL,
           MAX_CHARS=MAX_CHARS_PER_CHUNK, DEFAULT_MLX_MODEL=DEFAULT_MLX_MODEL,
           POCKET_MODEL=MLX_MODELS["pocket"], KOKORO_MODEL=MLX_MODELS["kokoro"],
           KOKORO_VOICES_STR=", ".join(KOKORO_VOICES), DEFAULT_KOKORO_VOICE=DEFAULT_KOKORO_VOICE,
           MLX_SAMPLE_RATE=MLX_SAMPLE_RATE, BACKEND=DEFAULT_BACKEND))

def interactive_loop():
    print(f"🔊  Orpheus TTS  |  voice: {VOICE}  |  model: {MODEL}")
    print("    Type text and press Enter. Ctrl+C to quit.\n")
    while True:
        try:
            text = input("Text> ").strip()
            if text:
                speak(text)
        except KeyboardInterrupt:
            print("\nBye!")
            break

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Check for help flag
        if sys.argv[1] in ("--help", "-h"):
            show_help()
            sys.exit(0)

        # Parse arguments
        input_text = None
        output_file = None
        backend = DEFAULT_BACKEND
        mlx_model = DEFAULT_MLX_MODEL
        kokoro_voice = DEFAULT_KOKORO_VOICE
        text_parts = []
        i = 1

        while i < len(sys.argv):
            arg = sys.argv[i]

            if arg in ("--file", "-f") and i + 1 < len(sys.argv):
                # Read from file
                file_path = sys.argv[i + 1]
                if not os.path.exists(file_path):
                    print(f"❌ File not found: {file_path}")
                    sys.exit(1)
                with open(file_path, 'r', encoding='utf-8') as f:
                    input_text = f.read().strip()
                if not input_text:
                    print("❌ File is empty")
                    sys.exit(1)
                i += 2

            elif arg in ("--save", "-s") and i + 1 < len(sys.argv):
                # Save to file
                output_file = sys.argv[i + 1]
                i += 2

            elif arg in ("--backend", "-b") and i + 1 < len(sys.argv):
                # Set backend
                backend = sys.argv[i + 1].lower()
                if backend not in ("ollama", "mlx"):
                    print(f"❌ Invalid backend: {backend}. Must be 'ollama' or 'mlx'")
                    sys.exit(1)
                i += 2

            elif arg in ("--mlx-model", "-m") and i + 1 < len(sys.argv):
                # Set MLX model
                mlx_model = sys.argv[i + 1].lower()
                if mlx_model not in MLX_MODELS:
                    print(f"❌ Invalid MLX model: {mlx_model}. Available: {list(MLX_MODELS.keys())}")
                    sys.exit(1)
                i += 2

            elif arg in ("--kokoro-voice", "-v") and i + 1 < len(sys.argv):
                # Set Kokoro voice
                kokoro_voice = sys.argv[i + 1].lower()
                if kokoro_voice not in KOKORO_VOICES:
                    print(f"❌ Invalid Kokoro voice: {kokoro_voice}. Available: {KOKORO_VOICES}")
                    sys.exit(1)
                i += 2

            else:
                # Collect text parts (skip if we already have input from file)
                if input_text is None:
                    text_parts.append(arg)
                i += 1

        # If no file input, use collected text parts
        if input_text is None and text_parts:
            input_text = " ".join(text_parts)

        # Execute based on parsed arguments
        if input_text:
            if output_file:
                save_audio(input_text, output_file, backend=backend, mlx_model=mlx_model, kokoro_voice=kokoro_voice)
            else:
                speak(input_text, backend=backend, mlx_model=mlx_model, kokoro_voice=kokoro_voice)
        else:
            print("❌ No text provided")
            sys.exit(1)
    else:
        interactive_loop()
