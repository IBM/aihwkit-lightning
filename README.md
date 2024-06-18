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

## Triton [Coming soon]
AIHWKIT-Lightning can be accelerated using [triton](https://triton-lang.org/main/index.html). This generally only makes sense when your layer sizes are in the thousands and when you want to split the layer into multiple tiles (only across the input dimension is supported).
To enable triton for `AnalogConv2d` and `AnalogLinear`, either `export AIHWKIT_USE_TRITON="1"` or execute your script as such `AIHWKIT_USE_TRITON="1" python your_script.py`. This feature is off by default.

## Notes
- Gradient behavior for `float16` and `bfloat16` does not match the AIHWKIT 100% due to rounding errors. This doesn't effect
training though and the gradient tests are passing for `float32` at `atol=1e-5`.
- Currently, `torch.compile` doesn't work when input range learning is activated.