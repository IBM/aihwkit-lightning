User Guide
============

Here, we will discuss the HW-Aware training capabilities of AIHWKIT-Lightning.

The so called :class:`aihwkit_lightning.simulator.configs.configs.TorchInferenceRPUConfig` is
defined in the :py:mod:`aihwkit_lightning.simulator.configs` module.

Typically, a user starts by defining a neural network using PyTorch:

.. code-block:: python
    
    from torch import nn
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

Then, one would like to convert the model to an anlog model using :func:`aihwkit_lightning.nn.conversion.convert_to_analog`.
For this, we need to define the Resistive Processing Unit (RPU) configuration from :class:`aihwkit_lightning.simulator.configs.configs.TorchInferenceRPUConfig`.

The parameters of the RPU configuration are:

- `forward` :class:`aihwkit_lightning.simulator.parameters.io.IOParameters`
- `clip` :class:`aihwkit_lightning.simulator.parameters.inference.WeightClipParameter`
- `modifier` :class:`aihwkit_lightning.simulator.parameters.inference.WeightModifierParameter`
- `mapping` :class:`aihwkit_lightning.simulator.parameters.mapping.MappingParameter`
- `pre_post` :class:`aihwkit_lightning.simulator.parameters.pre_post.PrePostProcessingParameter`

Let us begin by defining the `forward` parameter:

.. code-block:: python

    from aihwkit_lightning.nn.conversion import convert_to_analog
    from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
    rpu_config = TorchInferenceRPUConfig()
    rpu_config.forward.inp_res = 2**8 - 2
    rpu_config.forward.out_noise = 0.01
    rpu_config.forward.out_noise_per_channel = True
    rpu_config.forward.out_bound = 12.0
    rpu_config.forward.out_res = 2**8 - 2

What do those values mean? Let's break it down:

Input quantization
------------------

In AIHWKIT-Lightning, the inputs are quantized using bounds that are fixed at inference time, but can be learned
during training. The `inp_res` parameter defines the resolution of the input quantization. The operation can be
shown as this 

.. math::
    \begin{align} \label{eq:input-quant}
        & \mathbf{x^\text{quant}} \leftarrow \frac{\beta^\text{inp. quant}}{2^{\text{input bits}-1}-1} \cdot \lfloor \mathtt{clamp}(\mathbf{x},-\beta^\text{inp. quant},\beta^\text{inp. quant}) \cdot \frac{2^{\text{input bits}-1}-1}{\beta^\text{inp. quant}} \rceil
    \end{align}

The values :math:`\beta^\text{inp. quant}` are learnable variables defined per tile (one layer can be chunked into multiple tiles along the input dimension). If we don't chunk, we have one tile per layer.
In our example, we did :code:`rpu_config.forward.inp_res = 2**8 - 2`, meaning that we want to use 8-bit input quantization. This effectively results in having values in :math:`[-127, 127]` for the input quantization.

Ouput Noise
-----------

We also make use of output noise. This noise is injected after the MVM, and before the values are fed into the Analog to Digital Converter (ADC) (if the ADC is defined).

.. math::
    \begin{align} \label{eq:out-noise}
        & \mathbf{y^\text{noisy}}_{:,i} \leftarrow  \mathbf{y}_{:,i} + \mathbf{\kappa_i} \\
        & \mathbf{\kappa_i} = \nonumber
        \begin{cases}
            \gamma_\text{out} \cdot \beta^\text{inp. quant} \cdot \mathtt{max}(\mathtt{abs}(\mathbf{W}_{:,i})) \cdot \tau & \text{if } \mathtt{forward.out\_noise.out\_noise\_per\_channel} \\
            \gamma_\text{out} \cdot \beta^\text{inp. quant} \cdot \mathtt{max}(\mathtt{abs}(\mathbf{W})) \cdot \tau & \text{else} \\
            \text{where } \tau \sim \mathcal{N}(\mathbf{0},\mathbf{I}) \\
        \end{cases}
    \end{align}

here, :math:`\gamma_\text{out}` is the parameter we define using :code:`rpu_config.forward.out_noise`. You can now also see the effect of :code:`rpu_config.forward.out_noise_per_channel`.

ADC
---

The values we have defined so far make sense and we don't necessarily need an ADC. We still defined one here so that you can see how it works.
We model the ADC as a simple clipping operation followed by a quantization operation. The clipping operation is defined by the :code:`out_bound` parameter,
and the quantization operation is defined by the :code:`out_res` parameter.

.. math::
    \begin{align} \label{eq:out-quant}
        & \mathbf{y}^\text{quant}_i \leftarrow \mathtt{clamp}(\frac{{\beta}^\text{adc quant}_i}{2^{\text{adc bits}-1}-1} \cdot \lfloor{ \mathbf{y}_i \cdot \frac{2^{\text{adc bits}-1}-1}{{\beta}^\text{adc quant}_i}} \rceil, -{\beta}^\text{adc quant}_i, {\beta}^\text{adc quant}_i) \\
        & {\beta}^\text{adc quant}_i = \nonumber
        \begin{cases}
            \lambda_\text{adc} \cdot \beta^\text{inp. quant} \cdot \mathtt{max(abs(}\mathbf{W}_{:,i})) & \text{if } \mathtt{WeightClipType.CHANNELWISE\_SYMMETRIC} \\
            \lambda_\text{adc} \cdot \beta^\text{inp. quant} \cdot \mathtt{max(abs(}\mathbf{W})) & \text{if } \mathtt{WeightClipType.LAYERWISE\_SYMMETRIC} \\
        \end{cases}
    \end{align}

Here, the fixed :code:`out_bound` parameter is defined by :math:`{\beta}^\text{adc quant}` and the :code:`out_res` parameter is defined by :math:`2^{\text{adc bits}-1}-1`.

Clipping
--------
When we train models, we typically restrain the dynamic range of the weights. This happens after every update to the weights and is handled by
the :class:`aihwkit_lightning.optim.analog_optimizer.AnalogOptimizer`. We can move on to define the `clip` parameter of the RPU Configuration.

.. code-block:: python

    from aihwkit_lightning.simulator.configs import WeightClipType
    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL
    rpu_config.clip.sigma = 2.5

We typically clip to a specific number of standard deviations from the mean of the weights:

.. math::
    \begin{align} \label{eq:clipping}
        & \mathbf{W^*}_{:,i} \leftarrow {\mathtt{clamp}}(\mathbf{W}_{:,i}, -\mathbf{\zeta_i}, \mathbf{\zeta_i}) \\
        & \mathbf{\zeta_i} = \nonumber
        \begin{cases}
            \alpha \cdot {\mathtt{std}}(\mathbf{W}_{:,i}) & \text{if } \mathtt{WeightClipType.LAYER\_GAUSSIAN\_PER\_CHANNEL} \\
            \alpha \cdot {\mathtt{std}}(\mathbf{W}) & \text{if } \mathtt{WeightClipType.LAYER\_GAUSSIAN} \\
        \end{cases}
    \end{align}

Here, :math:`\alpha` is the parameter we define using :code:`rpu_config.clip.sigma`. For :math:`\alpha`, we recommend values in :math:`[2.0,3.5]`.
You can now also see the effect of :code:`rpu_config.clip.type`, which controls whether we clip per column or per tensor.

.. note::
    Clipping the weights per-column is typically better because weight columns can have different statistics. When we
    clip per tensor using :code:`WeightClipType.LAYER_GAUSSIAN`, we might clip some columns too much.

    You might ask: Why do you even have that then?

    This is because when you map weights to conductances, you normalize them to values in :math:`[-1, 1]`.
    If you clip per column, you need to be able to perform an affine correction to the result you get from performing the MVM.
    This requires that you store those affine correction parameters on-chip, using more memory.

Weight Noise Injection
----------------------
During the forward pass, one can inject noise into the weights. This is done by defining the `modifier` parameter of the RPU Configuration.
We now define that part of the RPU Configuration:

.. code-block:: python

    from aihwkit_lightning.simulator.configs import WeightModifierType
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL_PER_CHANNEL
    rpu_config.modifier.std_dev = 0.05

This simply adds Gaussian noise which is scaled with respect to the magnitude of the weights:

.. math::
    \begin{align} \label{eq:weight-noise}
        & \mathbf{W^\text{noisy}}_{:,i} \leftarrow  \mathbf{W}_{:,i} + \mathbf{\eta_i} \\
        & \mathbf{\eta_i} = \nonumber
        \begin{cases}
            \gamma_\text{weight} \cdot {\mathtt{max}}({\mathtt{abs}}(\mathbf{W}_{:,i})) \cdot \tau & \text{if } \mathtt{WeightModifierType.ADD\_NORMAL\_PER\_CHANNEL} \\
            \gamma_\text{weight} \cdot {\mathtt{max}}({\mathtt{abs}}(\mathbf{W})) \cdot \tau & \text{if } \mathtt{WeightModifierType.ADD\_NORMAL} \\
            \text{where } \tau \sim \mathcal{N}(\mathbf{0},\mathbf{I}) & \\
        \end{cases}
    \end{align}

Once can again see that we make the distinction between per-column and per-tensor noise injection.

Mapping
-------
The mapping parameter defines how we split up a layer into multiple tiles. AIHWKIT-Lightning only supports tiles with
an "infinite" number of output columns. This is because for HW-Aware training, we don't benefit from splitting the output
across the output dimension. This would only make sense if we would integrate IR-Drop into the training process. We can therefore
only configure the maximum number of input rows per tile. This is done this way:

.. code-block:: python

    rpu_config.mapping.max_input_size = 512

Now our virtual tiles have 512 input rows.

.. note::
    Layers are split evenly across the input dimension. This means that if you have a layer with 1024 input rows and you set
    :code:`rpu_config.mapping.max_input_size = 512`, you will get two tiles. The first tile will have input rows 0-511 and the
    second tile will have input rows 512-1023.

Input Range Learning
--------------------
The input range learning is a feature that allows the network to learn the input quantization bounds. This is done by defining the `pre_post` parameter of the RPU Configuration.

.. code-block:: python

    rpu_config.pre_post.input_range.enable = True
    rpu_config.pre_post.input_range.init_from_data = 100
    rpu_config.pre_post.input_range.init_std_alpha = 3.0

The first line enables the input range learning. The second line defines the number of batches to use for the initialization of the input range.
When we initialize the input range from data, we compute the standard deviation of the input activations. We then multiply this standard deviation by the value of `init_std_alpha` to get the initial input range.
The input range is then updated using a moving average of the input range over the batches.

.. note::
    We can also set the :code:`init_from_data = 0` and initialize the input ranges with fixed values using :code:`rpu_config.pre_post.input_range.init_value = <some value>`.

Analog conversion
-----------------

We can now convert the model using the defined RPU configuration:

.. code-block:: python

    analog_model = convert_to_analog(model, rpu_config)

See :func:`aihwkit_lightning.nn.conversion.convert_to_analog` for more information.

Analog Optimizer
----------------

We can now define the analog optimizer:

.. code-block:: python

    from aihwkit_lightning.optim import AnalogOptimizer
    from torch.optim import SGD
    optimizer = AnalogOptimizer(SGD, analog_model.analog_layers(), analog_model.parameters(), lr=0.1)

.. warning::
    The AnalogOptimizer essentially just attaches a :code:`step_post_hook` to the optimizer. The hook just
    iterates ove the analog layers and calls :code:`analog_layer.clip_weights()`.
    This means that your weights are not clipped if you're framework uses a different optimizer internally.
    DeepSpeed does this for example. In Huggingface, this also happens when you don't pass an optimizer to the :code:`Trainer`.