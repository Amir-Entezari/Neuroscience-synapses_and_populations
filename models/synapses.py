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


