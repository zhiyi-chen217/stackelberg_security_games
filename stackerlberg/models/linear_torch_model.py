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


class LinearTorchModel(TorchModelV2, nn.Module):
    """Linear Torch Model without bias."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self._model_l1 = nn.Linear(int(np.product(obs_space.shape)), int(np.product(obs_space.shape)) // 2, bias=False)
        self._model_l2 = nn.Linear(int(np.product(obs_space.shape)) // 2, num_outputs, bias=False)
        self._value_l1 = nn.Linear(int(np.product(obs_space.shape)), int(np.product(obs_space.shape)) // 2, bias=False)
        self._value_l2 = nn.Linear(int(np.product(obs_space.shape)) // 2, 1, bias=False)
        self.activation = nn.GELU()
        if model_config.get("custom_model_config", {}).get("constant_init", False):
            torch.nn.init.constant_(self._model_l1.weight, 0.01)
            torch.nn.init.constant_(self._model_l2.weight, 0.01)
            torch.nn.init.constant_(self._value_l1.weight, 0.01)
            torch.nn.init.constant_(self._value_l2.weight, 0.01)
        else:
            initializer = normc_initializer(0.01)
            initializer(self._model_l1.weight)
            initializer(self._model_l2.weight)
            initializer = normc_initializer(0.01)
            initializer(self._value_l1.weight)
            initializer(self._value_l2.weight)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._obs = obs
        x = self._model_l1(obs)
        x = self.activation(x)
        action_prob = self._model_l2(x)
        return action_prob, state

    @override(TorchModelV2)
    def value_function(self):
        value = self._value_l1(self._obs)
        value = self.activation(value)
        value = self._value_l2(value)
        return value.squeeze(1)
