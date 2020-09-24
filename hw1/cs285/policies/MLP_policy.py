import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete: # all envs in this hw have continuous action spaces, so never used?
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            # mean_net is the network you use to output the mean of the
            # distribution of action, and log std is the std of that
            # distribution. so in forward function, you should return a
            # distribution that has the mean from the output of mean_net
            # and the scale from torch.exp(logstd).
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        # TODO return the action that the policy prescribes
        # Call self.forward
        obs = ptu.from_numpy(obs)
        obs = obs.view(-1, obs.shape[0]) # make 2d!
        ac = self.forward(obs)
        return ptu.to_numpy(ac)

    # update/train this policy
    # don't even implement this lol, MLPPolicy class is never used except
    # as parent class.
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        # Should call forward on self.mean_net, then construct a distribution
        # (look in piazza for type) using the result and the value of logstd.
        if not self.discrete:
            mean_tensor = self.mean_net.forward(observation)

        # Then you have two choices. Can return dist, or sample from it with
        # rsample()
        dist = torch.distributions.normal.Normal(mean_tensor, self.logstd)
        # The FloatTensor returned would have dimension N x A
        return dist.rsample()

#####################################################
#####################################################

# multilayer perceptron supervised learning
class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # adv_n, acs_labels_na, qvals are unused
        # actions = true action (taken by the expert). Need to convert to pytorch tensor
        # pred_actions we get from calling our forward function on the observations
        # don't use get_action. Use self.forward, which doesn't return a numpy
        # array (so gradients can indeed flow)
        pred_actions = self.forward(ptu.from_numpy(observations))

        actions = ptu.from_numpy(actions)


        # Don't forget to set optimizer.no_grad()???
        # TODO: update the policy and return the loss
        loss = self.loss(pred_actions, actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
