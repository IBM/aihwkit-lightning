# AIHWKIT Evaluation
In this example, a model is converted to an analog-equivalent represenation using the aihwkit-lightning. Then, it is exported to an aihwkit representation which supports evaluation using a statistical inference model. Programming noise is applied using `drift_analog_weights()`.

The example prints the model before and after the conversion (so the `Conv2d`/`Linear` to `AnalogConv2d`/`AnalogLinear` replacement is visible), and then evaluates the exported model against the floating point baseline at several inference times. Because the weights of the model are random, the reported numbers are deviations from that baseline, not accuracies. Programming and drift noise are stochastic, so every drift time is repeated a few times and the spread is reported as well. The printing and feedback helpers live in `reporting.py`, so that `evaluation.py` only shows the conversion and evaluation flow.

Note: To run this example, both aihwkit and aihwkit-lightning must be installed.
