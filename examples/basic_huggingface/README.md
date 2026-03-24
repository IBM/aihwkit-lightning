# Basic HuggingFace Example

Fine-tunes a RoBERTa model on the GLUE CoLA task using AIHWKIT-Lightning's analog layers with the HuggingFace Trainer.

## Setup & Run

```bash
cd examples/basic_huggingface
uv sync
uv run test_huggingface.py
```

## What it does

1. Loads `FacebookAI/roberta-base` and converts it to an analog model using `convert_to_analog`
2. Tokenizes the CoLA dataset
3. Trains using the HuggingFace `Trainer` with an `AnalogOptimizer` wrapping AdamW
4. Evaluates using Matthews correlation coefficient
