<div align="center">

<img src="https://github.com/jyvet/reachy-mini-skills/blob/main/img/reachy-mini-skills.png?raw=true)" width="400"/>

</div>

Modular components for building Reachy Mini applications. This library provides a comprehensive set of skills including speech-to-text (STT), text-to-speech (TTS), large language model (LLM) integration, vision, and robot control capabilities.

## Features

- **Speech-to-Text (STT)**: Multiple provider support (Unmute, Cartesia, [WhisperSocket](https://github.com/jyvet/whispersocket))
- **Text-to-Speech (TTS)**: Multiple provider support (Unmute, Cartesia)
- **LLM Integration**: Support for Ollama and Cerebras
- **Vision**: Face tracking (OpenCL) and visual processing (Ollama-based VL models)
- **Robot Control**: Movement, emotions, and breathing animations
- **Audio Management**: Recording, playback, and audio processing

## Installation

### Prerequisites

This library requires the PortAudio development headers on Linux to be installed on your system:

```bash
# Debian/Ubuntu
sudo apt install portaudio19-dev

# Fedora
sudo dnf install portaudio-devel

# macOS
brew install portaudio
```

### From git (recommended for Reachy Mini apps)

Add to your `pyproject.toml` dependencies:

```toml
dependencies = [
    "reachy-mini-skills @ git+https://github.com/jyvet/reachy-mini-skills.git@main",
]
```

Or install directly:

```bash
uv add git+https://github.com/jyvet/reachy-mini-skills.git@main
```

### From source (for development)

```bash
git clone https://github.com/jyvet/reachy-mini-skills.git
cd reachy-mini-skills
uv sync
```

### With optional dependencies

```bash
# Install with Cartesia support
uv add "reachy-mini-skills[cartesia] @ git+https://github.com/jyvet/reachy-mini-skills.git@main"

# Install with Cerebras support
uv add "reachy-mini-skills[cerebras] @ git+https://github.com/jyvet/reachy-mini-skills.git@main"

# Install all optional dependencies
uv add "reachy-mini-skills[all] @ git+https://github.com/jyvet/reachy-mini-skills.git@main"

# For local development
uv sync --all-extras
```

## Quick Start

### Category-based imports (recommended)

```python
# Speech (STT & TTS)
from reachy_mini_skills.speech import stt_cartesia, tts_cartesia

stt = stt_cartesia.create()
tts = tts_cartesia.create()

result = await stt.transcribe(audio_queue)
await tts.synthesize("Hello world!")

# LLMs
from reachy_mini_skills.llms import cerebras

llm = cerebras.create()
response = await llm.generate(messages)

# Vision
from reachy_mini_skills.vision import vl_ollama

vl = vl_ollama.create()
img = vl.capture_image(robot)
description = await vl.describe_image(session, img)

# Movement
from reachy_mini_skills.movement import controller, emotions

mc = controller.MovementController(robot)
emotion = emotions.get_emotion_for_sentiment("happy", emotions_data)
```

### Full application setup

```python
from reachy_mini_skills import ReachyMiniApp, Config, AppSettings

# Create configuration
config = Config()

# Create and run your application
class MyApp(ReachyMiniApp):
    async def run(self):
        # Your application logic here
        pass

app = MyApp(config)
await app.start()
```

## Configuration

The library uses a modular configuration system:

```python
from reachy_mini_skills import Config, STTConfig, TTSConfig, LLMConfig

# Create custom configuration
config = Config(
    stt=STTConfig(
        energy_threshold=0.01,
        silence_timeout=1.2,
    ),
    tts=TTSConfig(
        cartesia_voice_id="your-voice-id",
    ),
    llm=LLMConfig(
        cerebras_default_model="gpt-oss-120b",
    ),
)
```

## Available Categories

### Import Patterns

```python
# Pattern 1: Import from categories (recommended)
from reachy_mini_skills.speech import stt_cartesia, tts_cartesia
from reachy_mini_skills.llms import cerebras
from reachy_mini_skills.vision import vl_ollama
from reachy_mini_skills.movement import controller

stt = stt_cartesia.create()
tts = tts_cartesia.create()
llm = cerebras.create()
vl = vl_ollama.create()

# Pattern 2: Import classes directly
from reachy_mini_skills import CartesiaSTT, CartesiaTTS, CerebrasLLM

# Pattern 3: Use factory functions
from reachy_mini_skills import get_stt_provider, get_tts_provider, get_llm_provider

stt = get_stt_provider("cartesia", config)
tts = get_tts_provider("cartesia", config)
llm = get_llm_provider("cerebras", config)
```

### Speech (`reachy_mini_skills.speech`)

| Module | Class | Description |
|--------|-------|-------------|
| `stt_whispersocket` | `WhisperSocketSTT` | [WhisperSocket](https://github.com/jyvet/whispersocket) WebSocket-based STT |
| `stt_cartesia` | `CartesiaSTT` | Cartesia speech recognition |
| `stt_unmute` | `UnmuteSTT` | Unmute.ai speech recognition |
| `tts_cartesia` | `CartesiaTTS` | Cartesia text-to-speech |
| `tts_unmute` | `UnmuteTTS` | Unmute.ai text-to-speech |

### LLMs (`reachy_mini_skills.llms`)

| Module | Class | Description |
|--------|-------|-------------|
| `cerebras` | `CerebrasLLM` | Cerebras cloud inference |
| `ollama` | `OllamaLLM` | Local Ollama models |

### Vision (`reachy_mini_skills.vision`)

| Module | Class | Description |
|--------|-------|-------------|
| `vl_ollama` | `VisionManager` | Ollama vision language model |

### Movement (`reachy_mini_skills.movement`)

| Module | Components | Description |
|--------|------------|-------------|
| `controller` | `MovementController`, `BreathingMove`, `TalkingMove` | Robot movement control |
| `emotions` | `load_emotions`, `get_emotion_for_sentiment` | Emotion management |
| `face_tracking` | `FaceTrackingWorker` | Face detection and tracking |

## Environment Variables

Set these environment variables for cloud providers:

```bash
export CARTESIA_API_KEY="your-cartesia-api-key"
export CEREBRAS_API_KEY="your-cerebras-api-key"
```

## Development

### Setup development environment

```bash
uv sync --all-extras
```

### Run tests

```bash
uv run pytest
```

### Format code

```bash
uv run black src tests
uv run ruff check src tests --fix
```

### Type checking

```bash
uv run mypy src
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
