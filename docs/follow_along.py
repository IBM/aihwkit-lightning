from torch import nn
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

from aihwkit_lightning.nn.conversion import convert_to_analog
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
rpu_config = TorchInferenceRPUConfig()
rpu_config.forward.inp_res = 2**8 - 2
rpu_config.forward.out_noise = 0.01
rpu_config.forward.out_noise_per_channel = True
rpu_config.forward.out_bound = 12.0
rpu_config.forward.out_res = 2**8 - 2

from aihwkit_lightning.simulator.configs import WeightClipType
rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL
rpu_config.clip.sigma = 2.5

from aihwkit_lightning.simulator.configs import WeightModifierType
rpu_config.modifier.type = WeightModifierType.ADD_NORMAL_PER_CHANNEL
rpu_config.modifier.std_dev = 0.05

rpu_config.mapping.max_input_size = 512

rpu_config.pre_post.input_range.enable = True
rpu_config.pre_post.input_range.init_std_alpha = 3.0
rpu_config.pre_post.input_range.init_from_data = 100
rpu_config.pre_post.input_range.init_value = 2.34

analog_model = convert_to_analog(model, rpu_config)

from aihwkit_lightning.optim import AnalogOptimizer
from torch.optim import SGD
optimizer = AnalogOptimizer(SGD, analog_model.analog_layers(), analog_model.parameters(), lr=0.1)