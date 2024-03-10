import torch

from aihwkit_lightning.nn import AnalogLinear
from aihwkit_lightning.nn.conversion import convert_to_analog


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def test_conversion():
    # Create a model
    model = Model()

    # Convert the model to analog
    analog_model = convert_to_analog(
        model,
        rpu_config=None,
        conversion_map=None,
        specific_rpu_config_fun=None,
        module_name="",
        ensure_analog_root=True,
        exclude_modules=None,
        inplace=False,
        verbose=False
    )

    # Check that the model has been converted to analog
    assert isinstance(analog_model.fc1, AnalogLinear)
    assert isinstance(analog_model.fc2, AnalogLinear)


if __name__ == "__main__":
    test_conversion()