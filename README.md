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