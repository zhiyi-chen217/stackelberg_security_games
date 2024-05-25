import numpy as np

# from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.visionnet import VisionNetwork as MyVisionNetwork
from ray.rllib.models.torch.misc import SlimConv2d, SlimFC
from ray.rllib.models.torch.misc import normc_initializer as normc_initializer
from ray.rllib.models.torch.misc import same_padding
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch

torch, nn = try_import_torch()


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        initializer = normc_initializer(0.01)
        initializer(m.weight)


class CNNPoacherTorchModel(TorchModelV2, nn.Module):
    """Linear Torch Model without bias."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        in_channels = 22
        out_channels_1 = 16
        out_channels_2 = 32
        self._obs_emd = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels_1, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=1, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(out_channels_2),
            nn.AdaptiveMaxPool2d((2, 1)),
            nn.Flatten())
        self._obs_emd.apply(weights_init)
        self._value =  nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels_1, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=3, padding=0, stride=1, bias=True),
            nn.BatchNorm2d(out_channels_2),
            nn.AdaptiveMaxPool2d((2, 1)),
            nn.Flatten(),
            nn.Linear(out_channels_2*2, 1))
        self._value.apply(weights_init)
        # if model_config.get("custom_model_config", {}).get("constant_init", False):
        #     torch.nn.init.constant_(self._value.weight, 0.01)
        # else:
        #     initializer = normc_initializer(0.01)
        #     initializer(self._value.weight)
        self._action_emd = nn.Sequential(nn.Linear(100, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, 64))
        self._action_emd.apply(weights_init)
        self._model = nn.Sequential(nn.Linear(128, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, num_outputs))
        self._model.apply(weights_init)
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["original_obs"]
        self._obs = obs
        observed_action = input_dict["obs"]["observed_action"]
        n_batch, n_row, n_col, n_channel = obs.shape
        obs = obs.reshape((n_batch, n_channel, n_row, n_col))
        obs_embedding = self._obs_emd(obs)
        action_embedding = self._action_emd(observed_action)
        embedding = torch.concat([obs_embedding, action_embedding], dim=1)
        output = self._model(embedding)
        return output, state

    @override(TorchModelV2)
    def value_function(self):
        n_batch, n_row, n_col, n_channel = self._obs.shape
        value = self._value(self._obs.reshape((n_batch, n_channel, n_row, n_col)))
        return value.squeeze(1)
