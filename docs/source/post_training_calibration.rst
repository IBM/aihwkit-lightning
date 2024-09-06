Post-Training Calibration Guide
===============================

When you have trained a mdoel without input range learning, you might still want to do input quantization.
Input quantization can be turned on by setting :code:`rpu_config.forward.inp_res = 2**8 - 2` (as an example for 8 bit input quantization).

For this, you need to calibrate the input ranges for each layer. These input ranges clip the input values to the desired range before
they are quantized. This calibration can be done using :func:`aihwkit_lightning.inference.calibration.calibration.calibrate_input_ranges`.

This function takes as input:

- Analog model (it must have been converted using :func:`aihwkit_lightning.nn.conversion.convert_to_analog`).
- Calibration type (see :class:`aihwkit_lightning.inference.calibration.calibration.InputRangeCalibrationType`).
- The dataloader to use for calibration.
- The quantile if the calibration type is :code:`InputRangeCalibrationType.CACHE_QUANTILE` or :code:`InputRangeCalibrationType.MOVING_QUANTILE`.
- The max samples controls the number of activation vectors we want to cache per layer for the calibration.
- The :code:`std_alpha` which controls how many standard deviations we want to use for the input ranges when the calibration type is :code:`InputRangeCalibrationType.MOVING_STD`.

.. note::
    The calibration types that use "MOVING" do not cache activations and are much faster and more memory efficient.
    The fastest method is :code:`InputRangeCalibrationType.MAX` but it might cause very large input ranges due to outliers.


.. warning::
    The dataloader you provide will be run through until the end. Make sure you provide a dataloader that does not run forever.

The following shows an example of a sampler that can be passed to the calibration function.

.. code-block:: python

    class Sampler:
    """Example of a sampler used for calibration."""

    def __init__(self):
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < total_num_samples:
            x = all_inputs[self.idx]
        else:
            raise StopIteration
        self.idx += 1
        if isinstance(linear_or_conv, AnalogConv2d):
            return (x,), {}

        return (), {"inp": x}

