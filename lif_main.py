from pymonntorch import *
import torch

import matplotlib.pyplot as plt

from models.dendrites import Dendrite
from models.currents import ConstantCurrent, NoisyCurrent, SetCurrent, StepFunction
from models.neurons import LIF
from models.synapses import SimpleSynapse, FullyConnectedSynapse, RandomConnectedFixedProbSynapse, \
    RandomConnectedFixedInputSynapse
from models.time_res import TimeResolution

# %%
net = Network(behavior={1: TimeResolution(dt=1.0), })

pop1 = NeuronGroup(
    net=net,
    size=100,
    behavior={
        # 2: SetCurrent(value=100),
        3: ConstantCurrent(value=7,
                           noise_range=7),
        4: Dendrite(),
        5: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-32,
            R=5,
            v_init="normal(-60, 10)",
        ),
        8: Recorder(variables=["u", "I"], tag="ng1_rec"),
        9: EventRecorder(variables=['spike'], tag="ng1_event")
    }
)
pop2 = NeuronGroup(
                    net=net,
                    size=100,
                    behavior={
                        # 2: SetCurrent(value=100),
                        3: ConstantCurrent(value=7),
                        4: Dendrite(),
                        5: LIF(
                            tau=10,
                            u_rest=-65,
                            u_reset=-70,
                            threshold=-32,
                            R=5,
                            # v_init="normal(-60, 10)",
                        ),
                        8: Recorder(variables=["u", "I"], tag="ng2_rec"),
                        9: EventRecorder(variables=['spike'], tag="ng2_event")
                    }
                )
syn = SynapseGroup(net=net,
                   tag="All",
                   src=pop1,
                   dst=pop2,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=10, sigma=1, n=10, variance=0.3)})

net.initialize()
net.simulate_iterations(iterations=100)

# Plot the current
plt.plot(net["ng2_rec", 0].variables["I"][:, :])
plt.xlabel('I(t)')
plt.ylabel('t')
plt.title('Current')
plt.show()
# Plot the membrane potential(voltage)
plt.plot(net["ng2_rec", 0].variables["u"][:, :])
plt.xlabel('U(t)')
plt.ylabel('t')
plt.title('Membrane Potential')
plt.show()

# Plot the raster plot
spike_events = net["ng2_event", 0].variables["spike"]
spike_times = spike_events[:, 0]
neuron_ids = spike_events[:, 1]
plt.figure(figsize=(8, 6))
plt.scatter(spike_times, neuron_ids, alpha=0.5)
plt.xlabel('Time')
plt.ylabel('Neuron ID')
plt.title('Raster Plot for LIF model')
plt.show()
