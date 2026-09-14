# AIHWKIT Evaluation

Converts a ResNet-32 to an analog-equivalent representation with AIHWKIT-Lightning, exports it to AIHWKIT, and evaluates it with a statistical inference model. Programming noise and conductance drift are applied using `drift_analog_weights()`.

## Setup & Run

```bash
cd examples/aihwkit_evaluation
uv sync
uv run evaluation.py
```

## What it does

1. Builds a ResNet-32 and prints it, then converts it with `convert_to_analog` and prints it again, so the `Conv2d`/`Linear` to `AnalogConv2d`/`AnalogLinear` replacement is visible
2. Exports the analog model to AIHWKIT with `export_to_aihwkit` and attaches a `PCMLikeNoiseModel` and a `GlobalDriftCompensation`
3. Evaluates the exported model against the floating point baseline at several inference times

## Notes

- The weights of the model are random (it is not trained), so the reported numbers are deviations from the floating point baseline, not accuracies
- Programming and drift noise are stochastic, so every drift time is repeated a few times and the spread is reported as well
- The printing and feedback helpers live in `reporting.py`, so that `evaluation.py` only shows the conversion and evaluation flow
- AIHWKIT ships wheels for CPython 3.10-3.12 only, hence the `requires-python = ">=3.10,<3.13"` in `pyproject.toml`; on newer interpreters it would have to be built from source
