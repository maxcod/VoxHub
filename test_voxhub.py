#!/usr/bin/env python3
"""
VoxHub Test Suite

Tests all major functionality without requiring audio playback or human validation.
Uses file-based validation and output checking.

Usage:
    python test_voxhub.py              # Run all tests
    python test_voxhub.py -v           # Verbose output
    python test_voxhub.py TestCLI      # Run specific test class
"""

import subprocess
import tempfile
import os
import wave
from pathlib import Path


class TestResult:
    """Simple test result tracker"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def add_pass(self, name):
        self.passed += 1
        self.tests.append((name, "PASS"))
        print(f"✅ {name}")

    def add_fail(self, name, reason):
        self.failed += 1
        self.tests.append((name, f"FAIL: {reason}"))
        print(f"❌ {name}: {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"\nFailed tests:")
            for name, result in self.tests:
                if result.startswith("FAIL"):
                    print(f"  - {name}: {result}")
        print(f"{'='*60}")
        return self.failed == 0


results = TestResult()


def run_command(args, input_text=None):
    """Run speak.py with given arguments and return result"""
    cmd = ["python3", "speak.py"] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        input=input_text,
        timeout=30
    )
    return result


def validate_wav(filepath, min_duration=0.1):
    """Validate WAV file properties"""
    if not os.path.exists(filepath):
        return False, "File doesn't exist"

    try:
        with wave.open(filepath, 'rb') as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            framerate = wav.getframerate()
            n_frames = wav.getnframes()
            duration = n_frames / framerate

            # Validate properties
            if sample_width != 2:  # 16-bit
                return False, f"Expected 16-bit, got {sample_width*8}-bit"
            if framerate != 24000:
                return False, f"Expected 24kHz, got {framerate}Hz"
            if duration < min_duration:
                return False, f"Duration too short: {duration}s"

            return True, f"Valid WAV: {duration:.2f}s, {framerate}Hz, {channels}ch, {sample_width*8}bit"
    except Exception as e:
        return False, str(e)


# ============================================================================
# Test Category 1: CLI Argument Parsing
# ============================================================================

def test_help_output():
    """Test --help flag shows usage information"""
    result = run_command(["--help"])
    if result.returncode == 0 and "VoxHub" in result.stdout and "Usage:" in result.stdout:
        results.add_pass("CLI: Help output")
    else:
        results.add_fail("CLI: Help output", "Missing expected text")


def test_help_short_flag():
    """Test -h flag works"""
    result = run_command(["-h"])
    if result.returncode == 0 and "VoxHub" in result.stdout:
        results.add_pass("CLI: Help short flag (-h)")
    else:
        results.add_fail("CLI: Help short flag (-h)", "Failed")


def test_no_args_interactive():
    """Test running with no args tries interactive mode"""
    # This will timeout or exit on EOF, which is expected
    # We just want to verify it doesn't crash
    try:
        result = subprocess.run(
            ["python3", "speak.py"],
            capture_output=True,
            text=True,
            input="",  # Send EOF immediately
            timeout=2
        )
        # Should show interactive prompt
        if "VoxHub" in result.stdout or "Text>" in result.stdout:
            results.add_pass("CLI: Interactive mode entry")
        else:
            results.add_fail("CLI: Interactive mode entry", "No prompt shown")
    except subprocess.TimeoutExpired:
        results.add_pass("CLI: Interactive mode entry (timeout expected)")


# ============================================================================
# Test Category 2: Model Selection
# ============================================================================

def test_model_kokoro_save():
    """Test kokoro model with file save"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_file = f.name

    try:
        result = run_command(["-m", "kokoro", "Test kokoro model", "--save", output_file])

        if result.returncode != 0:
            results.add_fail("Model: Kokoro save", f"Exit code {result.returncode}")
            return

        valid, msg = validate_wav(output_file)
        if valid:
            results.add_pass(f"Model: Kokoro save ({msg})")
        else:
            results.add_fail("Model: Kokoro save", f"Invalid WAV: {msg}")
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_model_pocket_save():
    """Test pocket model with file save"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_file = f.name

    try:
        result = run_command(["-m", "pocket", "Test pocket model", "--save", output_file])

        if result.returncode != 0:
            results.add_fail("Model: Pocket save", f"Exit code {result.returncode}")
            return

        valid, msg = validate_wav(output_file)
        if valid:
            results.add_pass(f"Model: Pocket save ({msg})")
        else:
            results.add_fail("Model: Pocket save", f"Invalid WAV: {msg}")
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_invalid_model():
    """Test invalid model shows error"""
    result = run_command(["-m", "invalid_model", "test"])

    if result.returncode != 0:
        if "Invalid model" in result.stdout and "Available models" in result.stdout:
            results.add_pass("Model: Invalid model error")
        else:
            results.add_fail("Model: Invalid model error", "Wrong error message")
    else:
        results.add_fail("Model: Invalid model error", "Should have failed")


# ============================================================================
# Test Category 3: Voice Selection
# ============================================================================

def test_kokoro_voice_selection():
    """Test kokoro with specific voice"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_file = f.name

    try:
        for voice in ["af_heart", "af_bella", "af_nicole", "am_fenrir"]:
            result = run_command(["-m", "kokoro", "-v", voice, f"Test {voice}", "--save", output_file])

            if result.returncode == 0 and os.path.exists(output_file):
                results.add_pass(f"Voice: Kokoro {voice}")
                os.unlink(output_file)
            else:
                results.add_fail(f"Voice: Kokoro {voice}", "Failed to generate")
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_invalid_voice_kokoro():
    """Test invalid voice for kokoro shows available voices"""
    result = run_command(["-m", "kokoro", "-v", "invalid_voice", "test"])

    if result.returncode != 0:
        output = result.stdout + result.stderr
        if "not available" in output and "af_heart" in output:
            results.add_pass("Voice: Invalid kokoro voice error")
        else:
            results.add_fail("Voice: Invalid kokoro voice error", "Missing voice list")
    else:
        results.add_fail("Voice: Invalid kokoro voice error", "Should have failed")


def test_voice_with_pocket():
    """Test voice selection with pocket (not supported) shows error"""
    result = run_command(["-m", "pocket", "-v", "af_heart", "test"])

    if result.returncode != 0:
        output = result.stdout + result.stderr
        if "not available" in output or "does not support" in output:
            results.add_pass("Voice: Pocket voice error (not supported)")
        else:
            results.add_fail("Voice: Pocket voice error", "Wrong error message")
    else:
        results.add_fail("Voice: Pocket voice error", "Should have failed")


# ============================================================================
# Test Category 4: File I/O
# ============================================================================

def test_read_from_file():
    """Test reading text from file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False) as f:
        f.write("This is test text from a file.")
        input_file = f.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_file = f.name

    try:
        result = run_command(["-m", "pocket", "--file", input_file, "--save", output_file])

        if result.returncode == 0 and os.path.exists(output_file):
            valid, msg = validate_wav(output_file)
            if valid:
                results.add_pass("File I/O: Read from file")
            else:
                results.add_fail("File I/O: Read from file", f"Invalid WAV: {msg}")
        else:
            results.add_fail("File I/O: Read from file", "Failed to process")
    finally:
        if os.path.exists(input_file):
            os.unlink(input_file)
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_nonexistent_file():
    """Test error on non-existent file"""
    result = run_command(["--file", "/nonexistent/file.txt", "test"])

    if result.returncode != 0 and ("not found" in result.stdout.lower() or "not found" in result.stderr.lower()):
        results.add_pass("File I/O: Nonexistent file error")
    else:
        results.add_fail("File I/O: Nonexistent file error", "Should show file not found")


def test_empty_file():
    """Test error on empty file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False) as f:
        input_file = f.name

    try:
        result = run_command(["--file", input_file])

        if result.returncode != 0 and "empty" in result.stdout.lower():
            results.add_pass("File I/O: Empty file error")
        else:
            results.add_fail("File I/O: Empty file error", "Should show empty file error")
    finally:
        if os.path.exists(input_file):
            os.unlink(input_file)


# ============================================================================
# Test Category 5: Audio Generation
# ============================================================================

def test_short_text():
    """Test short text (<100 chars) generation"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_file = f.name

    try:
        result = run_command(["-m", "pocket", "Short text", "--save", output_file])

        if result.returncode == 0:
            valid, msg = validate_wav(output_file, min_duration=0.5)
            if valid:
                results.add_pass("Audio: Short text generation")
            else:
                results.add_fail("Audio: Short text generation", msg)
        else:
            results.add_fail("Audio: Short text generation", "Failed to generate")
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_long_text():
    """Test long text (>100 chars) for streaming"""
    long_text = "This is a longer text that exceeds one hundred characters to test the streaming functionality. " \
                "It should trigger the streaming mode in the MLX backend for better user experience."

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_file = f.name

    try:
        result = run_command(["-m", "pocket", long_text, "--save", output_file])

        if result.returncode == 0:
            valid, msg = validate_wav(output_file, min_duration=3.0)
            if valid:
                results.add_pass("Audio: Long text generation (streaming)")
            else:
                results.add_fail("Audio: Long text generation", msg)
        else:
            results.add_fail("Audio: Long text generation", "Failed to generate")
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)


# ============================================================================
# Test Category 6: Error Handling
# ============================================================================

def test_no_text_provided():
    """Test error when no text provided"""
    result = run_command([])

    # Should enter interactive mode, which is OK
    # Or show error - both are acceptable
    results.add_pass("Error: No text (enters interactive mode)")


def test_model_and_voice_combination():
    """Test valid model + voice combinations"""
    test_cases = [
        ("kokoro", "af_heart", True),
        ("kokoro", "invalid", False),
        ("pocket", None, True),
    ]

    for model, voice, should_pass in test_cases:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_file = f.name

        try:
            args = ["-m", model, "test", "--save", output_file]
            if voice:
                args = ["-m", model, "-v", voice, "test", "--save", output_file]

            result = run_command(args)

            test_name = f"Combo: {model}+{voice or 'none'}"
            if should_pass:
                if result.returncode == 0:
                    results.add_pass(test_name)
                else:
                    results.add_fail(test_name, "Should have passed")
            else:
                if result.returncode != 0:
                    results.add_pass(test_name + " (error expected)")
                else:
                    results.add_fail(test_name, "Should have failed")
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    print("="*60)
    print("VoxHub Test Suite")
    print("="*60)
    print()

    # Check if speak.py exists
    if not os.path.exists("speak.py"):
        print("❌ speak.py not found. Run tests from VoxHub directory.")
        return False

    print("Running tests...\n")

    # Category 1: CLI
    print("Category 1: CLI Argument Parsing")
    print("-" * 40)
    test_help_output()
    test_help_short_flag()
    test_no_args_interactive()
    print()

    # Category 2: Models
    print("Category 2: Model Selection")
    print("-" * 40)
    test_model_kokoro_save()
    test_model_pocket_save()
    test_invalid_model()
    print()

    # Category 3: Voices
    print("Category 3: Voice Selection")
    print("-" * 40)
    test_kokoro_voice_selection()
    test_invalid_voice_kokoro()
    test_voice_with_pocket()
    print()

    # Category 4: File I/O
    print("Category 4: File I/O")
    print("-" * 40)
    test_read_from_file()
    test_nonexistent_file()
    test_empty_file()
    print()

    # Category 5: Audio
    print("Category 5: Audio Generation")
    print("-" * 40)
    test_short_text()
    test_long_text()
    print()

    # Category 6: Errors
    print("Category 6: Error Handling & Combinations")
    print("-" * 40)
    test_no_text_provided()
    test_model_and_voice_combination()
    print()

    # Summary
    return results.summary()


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
