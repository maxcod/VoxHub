# VoxHub Test Plan

## Testing Approach

### Strategy
Test the script without human audio validation by:
1. **Mocking audio playback** - Replace `sounddevice` with mock to verify calls
2. **File validation** - Generate WAV files and validate properties (size, format, duration)
3. **CLI testing** - Test argument parsing and error messages
4. **Output validation** - Capture stdout/stderr and validate messages

### What We Test
- ✅ Model selection (orpheus, pocket, kokoro)
- ✅ Voice selection and validation
- ✅ Backend auto-selection
- ✅ File I/O (reading text, saving audio)
- ✅ Error handling (invalid models, voices, files)
- ✅ Help output
- ✅ WAV file properties (sample rate, duration, format)

### What We Don't Test
- ❌ Audio quality (subjective, requires human listening)
- ❌ Actual model inference quality
- ❌ Network calls to Ollama (can mock if needed)

## Test Categories

### 1. CLI Argument Parsing Tests
- Default model selection
- Model flag parsing (-m, --model)
- Voice flag parsing (-v, --voice)
- File input parsing (-f, --file)
- Save output parsing (-s, --save)
- Help flag parsing (-h, --help)
- Combined flags

### 2. Model Selection Tests
- Test each model: orpheus, pocket, kokoro
- Invalid model shows error and available models
- Backend auto-selection works correctly

### 3. Voice Validation Tests
- Valid voices for orpheus (8 voices)
- Valid voices for kokoro (4 voices)
- Invalid voice for orpheus shows available voices
- Invalid voice for kokoro shows available voices
- Voice with pocket shows error (no voice support)

### 4. File I/O Tests
- Read text from file
- Non-existent file shows error
- Empty file shows error
- Save to WAV file
- Validate WAV file properties (format, sample rate)

### 5. Audio Generation Tests
- Short text (< 100 chars) - non-streaming
- Long text (> 100 chars) - streaming
- Verify WAV file is created
- Verify WAV duration matches expected
- Verify WAV format (16-bit PCM, 24kHz)

### 6. Error Handling Tests
- Invalid model name
- Invalid voice name
- Missing file
- Empty file
- No text provided

## Test Implementation Options

### Option 1: Python unittest with mocks (Recommended)
```python
import unittest
from unittest.mock import patch, MagicMock
import subprocess
import wave

class TestVoxHub(unittest.TestCase):
    def test_help_output(self):
        result = subprocess.run(['python', 'speak.py', '--help'],
                                capture_output=True, text=True)
        self.assertIn('VoxHub', result.stdout)

    def test_save_wav_file(self):
        # Test saving audio and validate WAV properties
        pass
```

### Option 2: Bash script tests
```bash
#!/bin/bash
# test_voxhub.sh
test_help() {
    output=$(python speak.py --help)
    [[ "$output" =~ "VoxHub" ]] || fail "Help output missing"
}
```

### Option 3: pytest (Most flexible)
```python
import pytest
from pathlib import Path

def test_model_selection():
    # Test model selection
    pass
```

## Recommended: pytest + WAV validation

### Advantages
- Clean test syntax
- Good fixtures support
- Easy mocking
- Can validate actual audio files
- No audio playback needed

### Test Structure
```
tests/
├── test_cli.py          # CLI argument tests
├── test_models.py       # Model selection tests
├── test_voices.py       # Voice validation tests
├── test_files.py        # File I/O tests
├── test_audio.py        # Audio generation tests
├── test_errors.py       # Error handling tests
└── fixtures/
    ├── short_text.txt   # < 100 chars
    └── long_text.txt    # > 100 chars
```

## Success Criteria

A test passes if:
1. ✅ Correct exit code (0 for success, non-zero for errors)
2. ✅ Expected output messages appear
3. ✅ WAV file created (for save tests)
4. ✅ WAV properties match expected (sample rate, format)
5. ✅ WAV duration approximately matches text length
6. ✅ No unexpected errors in stderr

## Quick Validation Method (Manual)

For quick testing without full framework:

```bash
# Test 1: Help
python speak.py --help | grep "VoxHub"

# Test 2: Invalid model
python speak.py -m invalid "test" 2>&1 | grep "Available models"

# Test 3: Invalid voice
python speak.py -m kokoro -v invalid "test" 2>&1 | grep "Available voices"

# Test 4: Save file
python speak.py -m pocket "test" --save /tmp/test.wav
file /tmp/test.wav | grep "WAVE audio"
```
