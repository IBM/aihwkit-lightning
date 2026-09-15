# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog], and this project adheres to
[Semantic Versioning]:

* `Added` for new features.
* `Changed` for changes in existing functionality.
* `Deprecated` for soon-to-be removed features.
* `Removed` for now removed features.
* `Fixed` for any bug fixes.
* `Security` in case of vulnerabilities.

## [2.1.0] - 2026-09-15

### Added
* `AnalogConv3d` layer. `torch.nn.Conv3d` modules are now converted by
  `convert_to_analog` and covered by the correctness tests against AIHWKIT
  (#39).
* Runtime dependencies (`torch`, `typing_extensions`, `tqdm`) and the optional
  extras `dev`, `huggingface` and `triton` are declared in `pyproject.toml`,
  together with a PEP 517 build backend (#28).
* PyPI classifiers in the project metadata (#66).
* `uv` project files, lockfiles and updated READMEs for the
  `basic_huggingface`, `deepspeed_and_huggingface` and `deepspeed_cifar10`
  examples.
* `examples/aihwkit_evaluation`: example that converts a Lightning model to
  AIHWKIT and evaluates it there, with a model summary and an evaluation
  report. Referenced from the user guide (#25).

### Changed
* `AnalogOptimizer` now takes a callable that returns the analog layers
  (e.g. `model.analog_layers`) instead of an already-created generator, so the
  post-step hook can iterate the layers on every optimizer step. Update calls
  from `AnalogOptimizer(SGD, model.analog_layers(), ...)` to
  `AnalogOptimizer(SGD, model.analog_layers, ...)`.
* Minimum supported Python version raised from 3.8 to 3.10.
* `torch` pinned to `2.13.0`. PyTorch 2.14 introduced C++ header changes that
  break the build (#68).
* `transformers` requirement raised to `>=5.10.0`. The HuggingFace examples
  were adapted to the v5 API, where `warmup_ratio` became `warmup_steps`
  (#77).
* Maintainers updated to Manuel Le Gallo-Bourdeau and Pablo Carmona Gonzalez
  (#80).
* Source file headers now reference the MIT license, matching `LICENSE.txt`
  and the package metadata. They previously still cited Apache 2.0 (#79).
* Read the Docs build moved to an Ubuntu LTS image and a Python version that
  matches the minimum requirement (#81).
* CI workflow runs on Python 3.14 (#21), pins the Triton commit and
  `setuptools` version, pins `torch`, `torchvision` and `torchaudio` for GCC
  compatibility (#68), and uses `actions/checkout` v7 (#64) and
  `actions/setup-python` v7 (#65).
* Type hints in the RNN module were tightened (`AnalogRNNCell` alias and
  explicit casts) to satisfy the newer `mypy` (#53).
* Development tooling bumps via Renovate: `black` 26.5.1, `mypy` 1.20.2,
  `pycodestyle` 2.14.0, `pytest` 9.x, `Sphinx` 9, `sphinx-rtd-theme` 3,
  `myst-parser` 5 and `recommonmark` 0.7.1.

### Fixed
* `WeightModifierType` is re-exported from `aihwkit_lightning.simulator.configs`
  again. It was dropped from that module in 2.0.1, which broke imports in
  code written for 1.x.
* `AnalogOptimizer` post-step hook silently stopped clipping weights after the
  first step because the generator of analog layers was exhausted.
* Type hints in the convolution modules.

### Security
* Refreshed `uv.lock` for the package and the examples to resolve Dependabot
  and Renovate vulnerability alerts (`transformers`, `black`, `pytest`,
  `gitpython`, `urllib3`, `pillow`, `setuptools`, `msgpack`, `accelerate`)
  (#42, #43, #54, #55, #67, #72, #73, #74, #75, #76, #77, #82, #83).

## [2.0.1] - 2025-06-26

### Added
* Post-training weight quantization. `quantize_weights()` on every analog
  layer and on `AnalogWrapper` quantizes the weights in place according to the
  configured `quantization_type` and then disables quantization and clipping
  in the `rpu_config` so that inference runs on the quantized weights (#23).
* `WeightNoiseInjectionType` (`NONE`, `ADD_NORMAL`, `ADD_NORMAL_PER_CHANNEL`)
  and `WeightQuantizationType` (`NONE`, `DISCRETIZE`, `DISCRETIZE_PER_CHANNEL`)
  enums. Noise injection and weight discretization are now configured
  independently via `rpu_config.modifier.noise_type` and
  `rpu_config.modifier.quantization_type`.
* `WeightClipType.LEARNABLE_PER_CHANNEL`: learnable per-channel weight clipping
  in the style of ParetoQ/LSQ. Analog layers hold a `learnable_weight_clip`
  parameter (initialized from the per-slice abs-max of the weights) and
  `UniformQuantize` returns a gradient with respect to the clipping range.
  Correctness against the ParetoQ reference is tested in `tests/test_methods.py`.
* `clip_and_quantize` and `sliced_abs_max` PyTorch utilities.
* GitHub Actions workflow "Formatting and tests" (`commit_check.yml`) running
  formatting, linting, type checks and the test suite.
* Triton kernels can be run and tested on CPU through the Triton interpreter
  (`TRITON_INTERPRET=1 TRITON_CPU_BACKEND=1`, wired into `make pytest`). The
  README documents how to build Triton-CPU.

### Changed
* Packaging moved from `setup.py`, `requirements.txt` and `scikit-build` to a
  `pyproject.toml` based on `setuptools`. The package version is now `2.0.1`
  and the license metadata is MIT.
* Weight quantization is applied at inference time as well, not only during
  training. Noise injection remains training-only.
* Models saved with 1.x are loaded transparently: at layer construction a
  legacy `rpu_config.modifier.type` is mapped onto `noise_type` and
  `quantization_type` with a warning (#24).
* `export_to_aihwkit` reads `modifier.noise_type` and always exports
  `enable_during_test=False`. `calibrate_input_ranges` disables noise through
  `noise_type`.
* Importing the package no longer fails on machines where Triton is installed
  but has no active driver (e.g. macOS). The corresponding `RuntimeError` is
  caught and the PyTorch path is used.
* Fewer parametrized cases in the Triton correctness tests.
* Read the Docs no longer installs `requirements.txt`.
* README: new cover image, GitHub Actions badge, nightly install marked as
  recommended, note that previous versions need `setuptools==75.1.0`.

### Deprecated
* `WeightModifierType` and `rpu_config.modifier.type`. Use `noise_type` and
  `quantization_type` instead.
* `rpu_config.modifier.enable_during_test`. It is forced to `False` with a
  warning.

### Removed
* Travis CI configuration (`.travis.yml`), replaced by GitHub Actions.
* `setup.py` and `requirements.txt`.

### Fixed
* Division by zero in `UniformQuantize` when a per-channel resolution is zero.
  The guard existed in 1.0.0 but was lost in 1.0.1 when the quantizer moved to
  `quant_utils.py`.

## [1.0.1] - 2025-05-23

### Added
* Triton kernels can run without CUDA: autotuning is disabled when no GPU is
  available (`lightning_autotune`) and explicit block sizes are passed
  instead. The Triton correctness tests fall back to CPU when CUDA is not
  present.
* `AIHWKIT_SKIP_TRITON=1` environment variable to skip the Triton tests
  (used by the Travis CPU job).
* Renovate configuration for automated dependency updates (#17).
* README sections describing the project, its key capabilities, the
  relationship to AIHWKIT and how to install a specific tagged version.

### Changed
* `UniformQuantize` moved from `torch_linear.py` to the new
  `nn/modules/torch_utils/quant_utils.py`.
* The `nn.modules` package uses relative imports.
* The forward argument of the convolution layers is named `inp` instead of
  `x_input`, matching `AnalogLinear`.

### Fixed
* `rpu_config` is deep-copied when it is written to and read from state-dict
  metadata, so loaded layers no longer share a config object with the
  checkpoint.

## [1.0.0] - 2025-04-22

### Added
* Analog layers `AnalogLinear`, `AnalogConv1d`, `AnalogConv2d`,
  `AnalogSequential` and `AnalogWrapper`, plus `AnalogRNN` with vanilla RNN,
  LSTM and GRU cells (uni- and bidirectional) ported from AIHWKIT (#13).
* `convert_to_analog` / `convert_to_digital` model conversion and the
  `AnalogOptimizer` wrapper that clips weights after every optimizer step.
* `TorchInferenceRPUConfig` covering: learnable input ranges with a fast
  I-BERT style mode (`fast_mode`, `act_range_momentum`) and dynamic per-input
  abs-max ranges (`dynamic`, #16); weight clipping (`WeightClipType`
  `LAYER_GAUSSIAN` and `LAYER_GAUSSIAN_PER_CHANNEL`); weight modifiers
  (`WeightModifierType` with additive Gaussian noise, discretization and
  per-channel variants); output noise; output quantization / ADC via
  `out_bound` and `out_res`; and tiling of large weight matrices across the
  input dimension via `max_input_size`.
* Triton mode (`AIHWKIT_USE_TRITON=1`): fused Triton kernels for the forward
  and backward pass of `AnalogLinear` and `AnalogConv2d`, including input
  range learning, weight modification and output noise, with autotuned block
  configurations.
* Input range calibration (`calibrate_input_ranges`).
* Export of trained models to AIHWKIT (`export_to_aihwkit`) (#3, #6).
* Examples for basic HuggingFace training, DeepSpeed with HuggingFace and
  DeepSpeed on CIFAR-10.
* Sphinx documentation published on Read the Docs, Travis CI (#2), citation
  file, and `MANIFEST.in` / package data so `VERSION.txt` ships with the
  package (#7, #8).
* Test suite: correctness against AIHWKIT, Triton correctness, conversion,
  export and speed benchmarks.

### Changed
* License changed from Apache 2.0 to MIT (2024-09-06).
* `move_to_meta` handled at the analog base layer level (#11).
* Lower memory footprint of input range learning (in-place clamping) and
  faster Triton kernels (output noise moved out of the inner loop).
* HuggingFace example uses `eval_strategy`; `transformers` pinned for the
  tests because newer versions broke `mypy` and `pylint`.

### Fixed
* Overflow in half precision during input range adaptation (#14).
* Weight quantization modified the weights in place and could divide by zero
  for a zero resolution (#15).
* Output bound handling and zero bounds in output quantization.
* Gradients did not flow through the input range because of a `no_grad`
  block.
* Fast-mode input range learning bug and DeepSpeed example fix.
* Division by zero when modifying weights.
* Per-column output noise and weight modifier reset bugs in the Triton
  kernels.
* LSTM state-dict handling and `AnalogConv1d` to/from digital conversion
  (#13).

## [0.0.1] - 2024-07-05
* Initial version with `triton` mode.


[2.1.0]: https://github.com/IBM/aihwkit-lightning/compare/v2.0.1...v2.1.0
[2.0.1]: https://github.com/IBM/aihwkit-lightning/compare/v1.0.1...v2.0.1
[1.0.1]: https://github.com/IBM/aihwkit-lightning/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/IBM/aihwkit-lightning/releases/tag/v1.0.0
[Keep a Changelog]: https://keepachangelog.com/en/1.0.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
