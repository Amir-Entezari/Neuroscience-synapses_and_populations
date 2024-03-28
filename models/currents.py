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


class NoisyCurrent(Behavior):
    def initialize(self, ng):
        self.iterations = self.parameter("iterations", None, required=True)
        self.noise_type = self.parameter("noise_type", "white")
        self.mean = self.parameter("mean", 0.0)
        self.std = self.parameter("std", 0.0)
        self.seed = self.parameter("seed", None)

        ng.I = ng.vector()

    def forward(self, ng):
        ng.I = ng.vector(float(self.add_noise(mean=self.mean,
                                              std=self.std,
                                              size=self.iterations)[ng.network.iteration]))

    def add_noise(self, mean, std, size):
        if self.seed is not None:
            self.set_seed()
        if self.noise_type == 'white':
            return self.white_noise(mean, std, size)
        elif self.noise_type == 'brownian':
            return self.brownian_noise(mean, std, size)
        else:
            raise ValueError("Unsupported noise type")

    def set_seed(self):
        np.random.seed(self.seed)

    def white_noise(self, mean, std, size):
        self.set_seed()
        return np.random.normal(mean, std, size)

    def brownian_noise(self, mean, std, size):
        # Generate white noise with mean=0 and std=1
        white_noise = np.random.normal(0, 1, size)

        # Generate cumulative sum to simulate Brownian motion
        brownian_motion = np.cumsum(white_noise)

        # Adjust mean and std
        adjusted_brownian_motion = (brownian_motion - np.mean(brownian_motion)) / np.std(brownian_motion)

        # Scale to desired mean and std
        scaled_brownian_noise = adjusted_brownian_motion * std + mean

        return scaled_brownian_noise


class StepFunction(Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value")
        self.t0 = self.parameter("t0")

        ng.I = ng.vector()


def forward(self, ng):
    if ng.network.iteration * ng.network.dt >= self.t0:
        ng.I += ng.vector(mode=self.value) * ng.network.dt
