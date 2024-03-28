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
        self.variance = self.parameter("variance", None, required=True)

        self.N = sg.src.size + sg.dst.size

        mean = self.j0 / self.N
        sg.W = sg.matrix(mode=f"normal({mean},{mean * self.variance})")
        # Make the diagonal zero
        if sg.src == sg.dst:
            sg.W.fill_diagonal_(0)
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
        sg.W = sg.matrix(mode=f"normal({mean},{mean * self.variance})")
        sg.W[torch.rand_like(sg.W) > self.p] = 0
        # Make the diagonal zero
        if sg.src == sg.dst:
            sg.W.fill_diagonal_(0)

        sg.I = sg.dst.vector()

    def forward(self, sg):
        pre_spike = sg.src.spike
        sg.I = torch.sum(sg.W[pre_spike], axis=0)


class RandomConnectedFixedInputSynapse(Behavior):
    """
    Random with fixed number of presynaptic partners connected synapse class that connect fixed number of neurons in
    a source and destination.
    """

    def initialize(self, sg):
        # Parameters:
        self.j0 = self.parameter("j0", None, required=True)
        self.variance = self.parameter("variance", None, required=True)
        self.n = self.parameter("n", None, required=True)

        self.N = sg.src.size + sg.dst.size

        mean = self.j0 / self.n
        sg.W = sg.matrix(mode=f"normal({mean},{mean * self.variance})")

        # Create a mask of which elements set to zero
        mask = self.create_random_zero_mask(sg.W.shape, self.n)
        # Apply the mask to the matrix
        sg.W[mask] = 0
        # Make the diagonal zero
        if sg.src == sg.dst:
            sg.W.fill_diagonal_(0)

        sg.I = sg.dst.vector()

    def forward(self, sg):
        pre_spike = sg.src.spike
        sg.I = torch.sum(sg.W[pre_spike], axis=0)

    def create_random_zero_mask(self, shape, n):
        mask = torch.full(shape, True, dtype=torch.bool)
        rows, cols = shape

        for col in range(cols):
            selected_rows = torch.randperm(rows)[:n]  # Randomly choose n indices for each column
            mask[selected_rows, col] = False  # Set selected elements to 1

        return mask  # Apply the mask to the original matrix

