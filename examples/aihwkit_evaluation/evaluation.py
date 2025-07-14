import torch
from model import resnet32
from aihwkit_lightning.simulator.configs import (
    TorchInferenceRPUConfig,
)
from aihwkit_lightning.nn.conversion import convert_to_analog
from aihwkit_lightning.nn.export import export_to_aihwkit
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation

if __name__ == "__main__":
    model = resnet32()
    rpu_config = TorchInferenceRPUConfig()
    model = convert_to_analog(model, rpu_config)
    aihwkit_model = export_to_aihwkit(model=model, max_output_size=-1)
    aihwkit_model.to(torch.float32)
    for analog_tile in aihwkit_model.analog_tiles():
        new_rpu_config = analog_tile.rpu_config
        break

    new_rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    new_rpu_config.drift_compensation = GlobalDriftCompensation()
    aihwkit_model.replace_rpu_config(new_rpu_config)
    aihwkit_model.eval()
    aihwkit_model.drift_analog_weights(0.)    