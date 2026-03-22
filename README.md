# VoxHub

Local TTS abstraction layer supporting multiple backends for privacy-focused voice synthesis.

## Features

- **Multiple Backends**: Ollama (Orpheus) and MLX-Audio (pocket-tts, Kokoro)
- **Voice Selection**: 8 Ollama voices + 4 Kokoro voices
- **Streaming Playback**: Audio starts playing immediately for long texts
- **File Export**: Save to WAV format
- **Local Inference**: All processing happens on your machine

## Supported Models

### Orpheus (Ollama Backend)
- **Model**: `legraphista/Orpheus:3b-ft-q8`
- **Backend**: Ollama (requires `ollama serve`)
- **Voices**: tara, leah, jess, leo, dan, mia, zac, zoe
- **Features**: Multi-voice support, chunked processing for long texts
- **Best for**: When Ollama is already running, or when you need Orpheus-specific voices

### Kokoro (MLX Backend - Apple Silicon)
- **Model**: `mlx-community/Kokoro-82M-bf16`
- **Backend**: MLX-Audio (Apple Silicon optimized)
- **Voices**:
  - `af_heart` (female, default)
  - `af_bella` (female)
  - `af_nicole` (female)
  - `am_fenrir` (male)
- **Features**: High-quality synthesis (82M parameters), streaming playback
- **Best for**: High-quality output with voice selection on Apple Silicon

### Pocket TTS (MLX Backend - Apple Silicon)
- **Model**: `mlx-community/pocket-tts-4bit`
- **Backend**: MLX-Audio (Apple Silicon optimized)
- **Voices**: None (no voice selection)
- **Features**: Fast generation, compact 4-bit quantized model
- **Best for**: Quick generation, when voice selection isn't needed

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd VoxHub
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

# Simple text-to-speech (uses kokoro model by default)
python speak.py "Hello world"

# Use Kokoro with specific voice
python speak.py -v af_bella "Hello world"

# Use Orpheus model
python speak.py -m orpheus "Hello world"

# Use pocket-tts (fast, no voice selection)
python speak.py -m pocket "Hello world"
```

### Read from File

```bash
python speak.py --file input.txt
python speak.py -m kokoro -v af_nicole --file input.txt
```

### Save to File

```bash
python speak.py "Your text" --save output.wav
python speak.py -m kokoro -v am_fenrir "Text" --save output.wav
```

### All Options

```
Options:
  --file, -f <path>    Read text from file
  --save, -s <path>    Save audio to WAV file
  --model, -m <name>   Model: "orpheus", "pocket", or "kokoro" (default: kokoro)
  --voice, -v <name>   Voice (model-specific)
  --help, -h           Show help message
```

## Configuration

Edit the `MODELS` dictionary in `speak.py` to modify default settings:

```python
DEFAULT_MODEL = "kokoro"  # Default model: orpheus, pocket, or kokoro
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama server URL
```

## Features

### Auto-Backend Selection
Backend is automatically selected based on the model:
- **orpheus** → Ollama backend (requires `ollama serve`)
- **pocket** → MLX backend (Apple Silicon)
- **kokoro** → MLX backend (Apple Silicon)

**Note:** On macOS with Apple Silicon, the MLX backend provides faster local inference.

### Streaming Playback
For texts > 100 characters, MLX models start playing audio immediately while generation continues.

### Voice Selection
- **orpheus**: 8 voices (tara, leah, jess, leo, dan, mia, zac, zoe)
- **kokoro**: 4 voices (af_heart, af_bella, af_nicole, am_fenrir)
- **pocket**: No voice selection

Invalid voice selection shows available options automatically

## Requirements

- Python 3.11+
- macOS with Apple Silicon (for MLX backend)
- Ollama (optional, for Ollama backend)

## Troubleshooting

### Invalid voice error
If you see "Voice 'x' not available for model 'y'", the error message will list all available voices for that model.

### Kokoro "words count mismatch" warnings
These warnings are harmless and don't affect audio quality. They indicate minor differences in tokenization.

### MLX model not loading
Ensure you have sufficient disk space (~1-2GB per model) and a stable internet connection for initial model download.

### Ollama connection errors
Make sure Ollama is running: `ollama serve`

### Invalid model error
Available models: orpheus, pocket, kokoro. Backend is auto-selected based on model choice.

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.
