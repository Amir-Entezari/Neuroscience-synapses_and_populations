from pymonntorch import *


class SetCurrent(Behavior):
    def initialize(self, ng):
        self.offset = self.parameter("value")
        ng.I = ng.vector(mode=self.offset)

    def forward(self, ng):
        ng.I.fill_(self.offset)


class ConstantCurrent(Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value", None, required=True)
        self.noise_range = self.parameter("noise_range", 0.0)
        ng.I = ng.vector(self.value)

    def forward(self, ng):
        ng.I = ng.vector(self.value)
        self.add_noise(ng)

    def add_noise(self, ng):
        ng.I += (ng.vector("uniform") - 0.5) * self.noise_range

