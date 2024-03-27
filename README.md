# AIHWKIT-Lightning ⚡

## Contributing
Install the development requirements.
```bash
pip install -r requirements_dev.txt
```
Create a branch from the `main` branch and make a well-documented PR. Make sure to run the following before submitting the PR:
```bash
make pytest
make black
make mypy
make pycodestyle
make pylint
```
All of these should pass.

## Notes
- Gradient behavior for `float16` and `bfloat16` does not match the AIHWKIT 100% due to rounding errors. This doesn't effect
training though and the gradient tests are passing for `float32` at `atol=1e-5`.
- Currently, `torch.compile` doesn't work when input range learning is activated.