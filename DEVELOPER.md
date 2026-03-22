# Developer Guide

Developer documentation for contributing to VoxHub.

## Development Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (required for Kokoro)
python -m spacy download en_core_web_sm
```

**Critical dependency:** `misaki[en]<0.9` - newer versions break Kokoro compatibility.

## Testing

### Running Tests

```bash
# Activate venv
source venv/bin/activate

# Run full test suite
python test_voxhub.py

# Expected: "Test Results: 21/21 passed"
```

### Test Coverage
- CLI argument parsing (3 tests)
- Model selection and validation (3 tests)
- Voice selection and validation (6 tests)
- File I/O operations (3 tests)
- Audio generation and WAV validation (2 tests)
- Error handling and combinations (4 tests)

**Note:** Tests validate functionality via file-based checks, not audio quality.

### Manual Testing

```bash
# Test help output
python speak.py --help

# Test invalid inputs (should fail gracefully with exit code 1)
python speak.py -m invalid "test"
python speak.py -m pocket -v test "test"
python speak.py -m kokoro -v invalid "test"

# Test audio generation (requires models)
python speak.py -m pocket "test" --save /tmp/test.wav
python speak.py -m kokoro -v af_bella "test" --save /tmp/test.wav
```

## Architecture

### Multi-Backend Design

VoxHub uses a **model-first API** where users select a model (`-m kokoro`) and the backend is automatically determined:

- **Ollama Backend**: Network API + SNAC decoder for token-to-audio
- **MLX Backend**: Local library for Apple Silicon optimized inference

Backend routing happens in `speak()` and `save_audio()` by checking `model_config["backend"]`.

### Voice Validation Philosophy

**Important:** Voice lists are NOT hardcoded in the `MODELS` dictionary. Voices are validated by the models themselves.

**Implementation:**
1. Check if model supports voices: `model_config["default_voice"] is None` means no voice support
2. If user provides voice for no-voice model → error immediately
3. Otherwise, pass voice directly to model and let model validate
4. Catch exceptions from model with try-except, surface errors to user

**Rationale:** Keeps voice lists automatically in sync with models, zero maintenance burden.

### Orpheus Token Processing

The Orpheus/Ollama backend uses a custom token-to-audio pipeline:

1. Text + voice → Format prompt: `<custom_token_3><|begin_of_text|>{voice}: {text}<|eot_id|>...`
2. Ollama returns strings like `<custom_token_N>`
3. Parse tokens → Convert to IDs: `token_id = raw_id - 10 - ((index % 7) * 4096)`
4. Group tokens in sets of 7 (must be divisible by 7)
5. Reshape into 3 SNAC codebooks
6. SNAC decoder → audio waveform

**Critical:** The index modulo operation in step 3 is essential for correct decoding.

### Streaming Playback

**MLX Streaming** (texts > 100 chars):
- Generator yields audio chunks from model
- Play each chunk sequentially with `sd.wait()`

**Ollama Chunking** (texts > 300 chars):
- Producer-consumer pattern with Queue (maxsize=2)
- Generation thread produces chunks
- Main thread consumes and plays
- Prevents memory buildup for very long texts

### Global State

- `snac_model`: Cached SNAC model (loaded once for Ollama backend)
- `mlx_models`: Dictionary cache of loaded MLX models (loaded once per model)

**Why:** Model loading is expensive (~1-2 seconds), cache to avoid reloading.

## Common Gotchas

### Exit Codes Matter
Error conditions MUST call `sys.exit(1)`, not just `return`. Tests check exit codes to validate error handling.

### No Global Config Variables
The old `VOICE`, `MODEL`, `SAMPLE_RATE` globals were removed. Functions now accept `model_path` and `sample_rate` as parameters. Get these from the `MODELS[model]` dict and pass explicitly.

### Voice Validation
Don't add voice lists back to the `MODELS` dictionary. Check `default_voice is None` for no-voice-support models, otherwise let the model validate.

### WAV Format
Output must be 16-bit PCM, 24kHz:
```python
audio_int16 = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)
wavfile.write(output_path, sample_rate, audio_int16)
```

### Timeout Handling
Ollama requests have 180s timeout for long generations.

## Project Structure

```
VoxHub/
├── speak.py           # All TTS logic (single-file design)
├── test_voxhub.py     # Test suite
├── requirements.txt   # Python dependencies
├── README.md          # User-facing documentation
├── DEVELOPER.md       # This file (developer documentation)
└── .gitignore         # Git ignore patterns
```

**Single-file design:** All logic in `speak.py` for simplicity.

## Adding a New Model

1. Add entry to `MODELS` dict in `speak.py`:
   ```python
   "new_model": {
       "backend": "mlx",  # or "ollama"
       "path": "hf-org/model-name",
       "default_voice": "voice_name",  # or None if no voice support
       "sample_rate": 24000,
   }
   ```

2. If new backend needed, implement:
   - `speak_BACKEND()` function
   - `save_audio_BACKEND()` function
   - Update routing in `speak()` and `save_audio()`

3. Add tests to `test_voxhub.py` for the new model

4. Update README.md with:
   - Model description
   - Example voices (if applicable)
   - "Best for" recommendation
   - Any special requirements

5. Run test suite - all tests must pass

## Contributing Workflow

1. Make your changes
2. Run test suite: `python test_voxhub.py` (all 21 tests must pass)
3. Test edge cases manually
4. Update README.md if user-facing changes
5. Update DEVELOPER.md if architecture changes
6. Commit and submit PR

## For Ollama Backend Development

```bash
# Start Ollama server
ollama serve

# Pull the Orpheus model
ollama pull legraphista/Orpheus:3b-ft-q8
```
