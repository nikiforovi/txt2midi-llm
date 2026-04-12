# Text-to-MIDI (txt2midi)

A baseline implementation for generating MIDI music from text prompts using a Transformer model.

## Features

- **REMI Tokenization**: Efficient MIDI-to-token encoding.
- **Prompt Encoder**: BERT-based text embedding for musical context.
- **Music Transformer**: Decoder-only architecture for sequence generation.
- **FastAPI Integration**: Simple HTTP API for generating MIDI files.

## Installation

Follow these steps to set up the environment:

```bash
# Install Poetry if you haven't already
# curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

## Usage

### 1. Data Preparation
Place your dataset (JSONL format) in `data/datasets/baseline.jsonl`.

### 2. Training
Start the training process:
```bash
poetry run python training/train.py
```

### 3. API & Inference
Run the server:
```bash
poetry run python api/server.py
```

Generate music via API:
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "chill pop piano in A major", "temperature": 0.8}' \
     --output song.mid
```

## Project Structure

- `tokenizer/`: MIDI tokenization logic.
- `model/`: Neural network architectures.
- `api/`: REST API.
- `configs/`: YAML configurations.

## License
GNU GPL v3.0
