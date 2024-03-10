from typing import Optional, Tuple, Type
from torch import Tensor
from torch.nn import Linear
from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig
from aihwkit_lightning.nn.modules.base import AnalogLayerBase

class AnalogLinear(Linear, AnalogLayerBase):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        rpu_config: TorchInferenceRPUConfig = None,
    ) -> None:
        Linear.__init__(self, in_features, out_features, bias, device, dtype)
        self.rpu_config = rpu_config
    
    @classmethod
    def from_digital(
        cls, module: Linear, rpu_config: TorchInferenceRPUConfig, tile_module_class: Optional[Type] = None
    ) -> "AnalogLinear":
        """Return an AnalogLinear layer from a torch Linear layer.

        Args:
            module: The torch module to convert. All layers that are
                defined in the ``conversion_map``.
            rpu_config: RPU config to apply to all converted tiles.
                Applied to all converted tiles.

        Returns:
            an AnalogLinear layer based on the digital Linear ``module``.
        """
        analog_layer = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
            rpu_config=rpu_config,
        )

        analog_layer.set_weights(module.weight.data, None if module.bias is None else module.bias.data)
        return analog_layer

    @classmethod
    def move_to_meta(cls, module: Linear):
        """Move the module to the meta class.

        This is used to move the module to the meta class. This is
        useful for the conversion of the module to analog.

        Args:
            module: The module to move to the meta class.

        """
        module.weight = module.weight.to(device="meta")
        if module.bias is not None:
            module.bias = module.bias.to(device="meta")

    def set_weights(self, weight: Tensor, bias: Optional[Tensor] = None) -> None:
        """Set the weight (and bias) tensors to the analog crossbar. Creates a copy of the tensors.

        Args:
            weight: the weight tensor
            bias: the bias tensor is available
        """
        assert self.weight.shape == weight.shape, f"weight shape mismatch. Got {weight.shape}, expected {self.weight.shape}"
        if bias is not None:
            assert self.bias.shape == bias.shape, f"bias shape mismatch. Got {bias.shape}, expected {self.bias.shape}"
        self.weight.data = weight.detach().clone()
        self.bias.data = bias.detach().clone()

    def get_weights(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Get the weight (and bias) tensors from the analog crossbar.

        Args:
            **kwargs: see tile level,
                e.g. :meth:`~aihwkit.simulator.tiles.analog.AnalogTile.get_weights`.

        Returns:
            tuple: weight matrix, bias vector

        Raises:
            ModuleError: if not of type TileModule.
        """
        return (self.weight, self.bias)