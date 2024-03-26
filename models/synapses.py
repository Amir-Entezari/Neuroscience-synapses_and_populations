import random

import torch
import numpy as np
from pymonntorch import Behavior


class SimpleSynapse(Behavior):
    def initialize(self, sg):
        self.mode = self.parameter("mode", "normal(1.0,0.0)")
        self.alpha = self.parameter("alpha", 1.0)
        sg.W = sg.matrix(mode=self.mode)
        sg.I = sg.dst.vector()

    def forward(self, sg):
        pre_spike = sg.src.spike
        sg.I = torch.sum(sg.W[pre_spike], axis=0) * self.alpha


class FullyConnectedSynapse(Behavior):
    """
    Fully connected synapse class that connect all neurons in a source and destination.
    """

    def initialize(self, sg):
        self.j0 = self.parameter("j0", None, required=True)
        self.sigma = self.parameter("sigma", None, required=True)

        self.N = sg.src.size + sg.dst.size

        mean = self.j0 / self.N
        variance = self.sigma / np.sqrt(self.N)
        sg.W = sg.matrix(mode=f"normal({mean},{variance})")

        sg.I = sg.dst.vector()

    def forward(self, sg):
        pre_spike = sg.src.spike
        sg.I = torch.sum(sg.W[pre_spike], axis=0)


class RandomConnectedFixedProbSynapse(Behavior):
    """
    Random connected with fixed coupling probability synapse class that connect neurons in a source and destination
    with a probability
    """

    def initialize(self, sg):
        # Parameters:
        self.j0 = self.parameter("j0", None, required=True)
        self.variance = self.parameter("variance", None, required=True)
        self.p = self.parameter("p", None, required=True)

        self.N = sg.src.size + sg.dst.size

        mean = self.j0 / (self.p * self.N)
        # variance = self.p * (1 - self.p) * self.N
        sg.W = sg.matrix(mode=f"normal({mean},{self.variance})")
        sg.W[torch.rand_like(sg.W) > self.p] = 0
        sg.I = sg.dst.vector()

    def forward(self, sg):
        pre_spike = sg.src.spike
        sg.I = torch.sum(sg.W[pre_spike], axis=0)


