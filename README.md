# VoxHub

Local TTS abstraction layer supporting multiple backends for privacy-focused voice synthesis.

## Features

- **Multiple Backends**: Ollama (Orpheus) and MLX-Audio (pocket-tts, Kokoro)
- **Voice Selection**: 8 Ollama voices + 4 Kokoro voices
- **Streaming Playback**: Audio starts playing immediately for long texts
- **File Export**: Save to WAV format
- **Local Inference**: All processing happens on your machine

## Supported Models

### Ollama Backend
- Model: `legraphista/Orpheus:3b-ft-q8`
- Voices: tara, leah, jess, leo, dan, mia, zac, zoe
- Features: Multi-voice, chunked processing

### MLX Backend (Apple Silicon)
- **pocket-tts-4bit**: Fast, compact model
- **Kokoro-82M-bf16**: High-quality model with 4 voices:
  - `af_heart` (female, default)
  - `af_bella` (female)
  - `af_nicole` (female)
  - `am_fenrir` (male)

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd orpheus-tts
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**Important for Kokoro**: The `misaki` package must be version <0.9:
```bash
pip install "misaki[en]<0.9"
```

### 4. Download spaCy model (for Kokoro)
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Simple text-to-speech
python speak.py "Hello world"

# Use MLX backend
python speak.py --backend mlx "Hello world"

# Use Kokoro model
python speak.py -b mlx -m kokoro "Hello world"

# Use specific Kokoro voice
python speak.py -b mlx -m kokoro -v af_bella "Hello world"
```

### Read from File

```bash
python speak.py --file input.txt
python speak.py -b mlx -m kokoro -v af_nicole --file input.txt
```

### Save to File

```bash
python speak.py "Your text" --save output.wav
python speak.py -b mlx -m kokoro -v am_fenrir "Text" --save output.wav
```

### All Options

```
Options:
  --file, -f <path>         Read text from file
  --save, -s <path>         Save audio to WAV file
  --backend, -b <name>      Backend: "ollama" or "mlx" (default: ollama)
  --mlx-model, -m <name>    MLX model: "pocket" or "kokoro" (default: pocket)
  --kokoro-voice, -v <name> Kokoro voice (default: af_heart)
  --help, -h                Show help message
```

## Configuration

Edit these variables in `speak.py`:

```python
# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "legraphista/Orpheus:3b-ft-q8"
VOICE = "tara"

# MLX settings
DEFAULT_MLX_MODEL = "pocket"
DEFAULT_KOKORO_VOICE = "af_heart"
```

## Features

### Streaming Playback
For texts > 100 characters, audio starts playing immediately while generation continues in the background.

### Multiple Backends
Switch between Ollama and MLX backends depending on your needs:
- **Ollama**: More voice options, requires Ollama server
- **MLX**: Faster on Apple Silicon, local inference

## Requirements

- Python 3.11+
- macOS with Apple Silicon (for MLX backend)
- Ollama (optional, for Ollama backend)

## Troubleshooting

### Kokoro "words count mismatch" warnings
These warnings are harmless and don't affect audio quality. They indicate minor differences in tokenization.

### MLX model not loading
Ensure you have sufficient disk space (~1-2GB per model) and a stable internet connection for initial model download.

### Ollama connection errors
Make sure Ollama is running: `ollama serve`

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.
