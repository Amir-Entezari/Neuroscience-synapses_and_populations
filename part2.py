# %% [markdown]
# # Project 2: part2

# %% [markdown]
# ## Import libraries

# %%
from pymonntorch import *
import torch

import matplotlib.pyplot as plt

from models.activity import ActivityRecorder
from models.dendrites import Dendrite
from models.currents import ConstantCurrent, NoisyCurrent, SetCurrent, CurrentSum, RandomCurrent
from models.neurons import LIF, ELIF, AELIF
from models.synapses import SimpleSynapse, FullyConnectedSynapse, RandomConnectedFixedProbSynapse, \
    RandomConnectedFixedInputSynapse
from models.time_res import TimeResolution
from simulate import Simulation

# %%
def plot_neuron_activity(net, ng_name, title, info_text=None):

    fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True)

    # Plot the membrane potential(voltage)
    axs[0, 0].plot(net[f"{ng_name}_rec", 0].variables["u"][:, :])
    axs[0, 0].axhline(y=net.NeuronGroups[0].behavior[5].init_kwargs['threshold'], color='red', linestyle='--', label=f"{ng_name} Threshold")
    axs[0, 0].set_xlabel('t')
    axs[0, 0].set_ylabel('U(t)')
    axs[0, 0].legend()
    axs[0, 0].set_title('Membrane Potential')

    # Plot the current
    axs[1, 0].plot(net[f"{ng_name}_rec", 0].variables["I"][:, :])
    axs[1, 0].set_xlabel('t')
    axs[1, 0].set_ylabel('I(t)')
    axs[1, 0].legend()
    axs[1, 0].set_title('Current')

    # Plot the raster plot
    spike_events = net[f"{ng_name}_event", 0].variables["spike"]
    spike_times = spike_events[:, 0]
    neuron_ids = spike_events[:, 1]
    axs[0, 1].scatter(spike_times, neuron_ids, s=5)
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Neuron ID')
    axs[0, 1].legend()
    axs[0, 1].set_title('Raster Plot for LIF model')

    # Plot the activity
    axs[1, 1].plot(net[f"{ng_name}_rec", 0].variables["activity"])
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('activity')
    axs[1, 1].legend()
    axs[1, 1].set_title('Activity')

    # Additional subplot for text and information
    axs[2, 0].axis('off')  # Turn off the axis for this subplot
    axs[2, 1].axis('off')
    axs[2, 0].text(0.1, 0.5, info_text, bbox=dict(facecolor='white', alpha=0.5))

    fig.suptitle(title, fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_weight_distribution(weight_matrix, bins=50, density=True):
    # Flatten the weight matrix into a 1D array
    weights_flat = weight_matrix.flatten()

    # Plot the distribution using a histogram
    plt.hist(weights_flat, bins=bins, density=density, alpha=0.7, color='b')
    plt.title('Weight Matrix Distribution')
    plt.xlabel('Weight Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()



# %% [markdown]
# # Part 2: Implementing 3 synapses connectivity schemes

# %% [markdown]
# The 3 synapses connectivity schemes has been implemented in synapses model, here we just use and experiment them with examples.

# %% [markdown]
# We have experimented the fully connected scheme in the part1, based on the goal of this part, here we focus more in the synapse parameters. we begin with a simple one population synapse:

# %% [markdown]
# # 1- Fully connected synapses

# %% [markdown]
# ## 1.1 Inside one neuron group synapse:
# We saw the behavior of the population when there is no noise or randomness in neurons; Now we change parameters of synapse to see results. For example, we test the population with a low variance(5% of the weights), normal variance(25% of the weights), high variance(50% of the weights) and extremely high variance(100% of the weights).

# %% [markdown]
# ### 1.1.1 Change in j0

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.6),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.6),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.6),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=0.25)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: FullyConnectedSynapse(j0=25, variance=0.25)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: FullyConnectedSynapse(j0=50, variance=0.25)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2, text_y=0.5)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2, text_y=0.5)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2, text_y=0.5)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-full-synapse-diff-j.pdf")
plt.show()

# %% [markdown]
# As we see, beside the fact that all neurons are the same at the begining, when we increase the variance of weights, the time of spikes of neurons will be different.

# %% [markdown]
# ### Random current

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: RandomCurrent(mean=6, std=2),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: RandomCurrent(mean=6, std=2),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: RandomCurrent(mean=6, std=2),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=0.25)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: FullyConnectedSynapse(j0=25, variance=0.25)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: FullyConnectedSynapse(j0=50, variance=0.25)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-full-synapse-diff-j-rand-curr.pdf")
plt.show()

# %% [markdown]
# ### 1.1.2 Change in variance

# %% [markdown]
# Noisy constant current

# %%
sim = Simulation(net=Network(behavior={0: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.6),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.6),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.6),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=0.1)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=0.50)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=0.9)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-full-synapse-diff-variance.pdf")
plt.show()

# %% [markdown]
# ### random current

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: RandomCurrent(mean=6, std=0.5),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: RandomCurrent(mean=6, std=0.5),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: RandomCurrent(mean=6, std=0.5),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=0.05)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=0.5)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=1)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-full-synapse-diff-variance-rand-curr.pdf")
plt.show()

# %% [markdown]
# 

# %% [markdown]
# ### Size of ng

# %% [markdown]
# #### COnstant and noisy current

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=50,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.5),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.5),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=250,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.5),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=0.25)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=0.25)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=0.25)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-full-synapse-diff-size.pdf")
plt.show()

# %% [markdown]
# 

# %% [markdown]
# #### Different neuron models:

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="LIF",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.5),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="LIF_rec"),
        10: EventRecorder(variables=['spike'], tag="LIF_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ELIF",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=13.8, noise_range=0.5),
        6: Dendrite(),
        7: ELIF(R=1.7,
                tau=10,
                threshold=-13,
                rh_threshold=-42,
                u_rest=-65,
                u_reset=-73,
                delta_T=0.1
                ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ELIF_rec"),
        10: EventRecorder(variables=['spike'], tag="ELIF_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="AELIF",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=30, noise_range=0.5),
        6: Dendrite(),
        7: AELIF(a=0.5,
                 b=7,
                 R=1.7,
                 tau_m=10,
                 tau_w=100,
                 threshold=-13,
                 rh_threshold=-42,
                 u_rest=-65,
                 u_reset=-73,
                 delta_T=0.1
                 ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="AELIF_rec"),
        10: EventRecorder(variables=['spike'], tag="AELIF_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="LIF-LIF",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=0.25)})
syn_exc2 = sim.add_synapse_group(
                   tag="ELIF-ELIF",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=0.25)})
syn_exc3 = sim.add_synapse_group(
                   tag="AELIF-AELIF",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=0.25)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-full-synapse-diff-neuron-model.pdf")
plt.show()

# %% [markdown]
# ## 1.2 Synapse between two groups:
# Now let's create a synapse between two groups of neurons. First, we test when there's no noise and all neurons begin with the same membrane potential. Now we change parameters of synapse to see results. For example, we test the population with a low variance(5% of the weights), normal variance(25% of the weights), high variance(50% of the weights) and extremely high variance(100% of the weights).

# %% [markdown]
# ### 1.2.1 Change in j0

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.5),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.5),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: FullyConnectedSynapse(j0=15, variance=0.2)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: FullyConnectedSynapse(j0=15, variance=0.2)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-full-synapse-noise-curr.pdf")
plt.show()

# %% [markdown]
# #### random curr

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: RandomCurrent(mean=6, std=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: RandomCurrent(mean=6, std=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=0.2)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: FullyConnectedSynapse(j0=5, variance=0.2)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-full-synapse-low-j-rand-curr.pdf")
plt.show()

# %% [markdown]
# different j

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: RandomCurrent(mean=6, std=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: RandomCurrent(mean=6, std=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: FullyConnectedSynapse(j0=100, variance=0.2)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: FullyConnectedSynapse(j0=10, variance=0.2)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-full-synapse-diff-j-rand-curr.pdf")
plt.show()

# %% [markdown]
# #### Change variance

# %% [markdown]
# noisy curr

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.5),
        # 2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.5),
        # 2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: FullyConnectedSynapse(j0=10, variance=0.9)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: FullyConnectedSynapse(j0=10, variance=0.9)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-full-synapse-high-variance-noise-curr.pdf")
plt.show()

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.5),
        2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.5),
        2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: FullyConnectedSynapse(j0=10, variance=0.9)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: FullyConnectedSynapse(j0=10, variance=0.9)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-full-synapse-high-variance-rand-curr.pdf")
plt.show()

# %% [markdown]
# Different variance

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.5),
        # 2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.5),
        # 2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: FullyConnectedSynapse(j0=10, variance=0.1)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: FullyConnectedSynapse(j0=10, variance=0.9)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-full-synapse-diff-variance-noise-curr.pdf")
plt.show()

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.5),
        # 2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.5),
        # 2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: FullyConnectedSynapse(j0=10, variance=100)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: FullyConnectedSynapse(j0=10, variance=0.9)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-full-synapse-very-high-variance-noise-curr.pdf")
plt.show()

# %% [markdown]
# #### Change in size

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=250,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.6),
        # 2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=50,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.6),
        # 2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: FullyConnectedSynapse(j0=10, variance=0.5)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: FullyConnectedSynapse(j0=10, variance=0.5)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-full-synapse-diff-size-noise-curr.pdf")
plt.show()

# %% [markdown]
# # 2.  Random coupling: Fixed coupling probability

# %% [markdown]
# ## One population

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc = sim.add_neuron_group(
    tag="ng_exc",
    size=250,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.0),
        # 2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc_event")
    }
)

syn = sim.add_synapse_group(
                   tag="exc-exc",
                   src=ng_exc,
                   dst=ng_exc,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=10, p=0.1, variance=0.25)})

sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(2, 2, figsize=(18, 8), sharex=True)
ng_exc.add_membrane_potential_plot(axs[0, 0], model_idx=7)
ng_exc.add_neuron_model_params_info(axs[0, 0], model_idx=7)

ng_exc.add_current_plot(axs[1, 0])
ng_exc.add_current_params_info(axs[1,0], current_idx=2)

ng_exc.add_raster_plot(axs[0, 1])
ng_exc.add_activity_plot(axs[1, 1])
fig.suptitle("Activity for one neuron group", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-prob-synapse-const-curr.pdf")
plt.show()

# %% [markdown]
# ### Change in curr

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.0),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.6),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: RandomCurrent(mean=6, std=0.2),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=10, p=0.1, variance=0.25)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=10, p=0.1, variance=0.25)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=10, p=0.1, variance=0.25)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2, text_y=0.5)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2, text_y=0.5)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2, text_y=0.5)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-prob-synapse-diff-curr.pdf")
plt.show()

# %% [markdown]
# ### Change in j

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.6),
        # 2: RandomCurrent(mean=6, std=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.6),
        # 2: RandomCurrent(mean=6, std=0.2),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.6),
        # 2: RandomCurrent(mean=6, std=0.2),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.25)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=25, p=0.1, variance=0.25)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=35, p=0.1, variance=0.25)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2, text_y=0.5)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2, text_y=0.5)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2, text_y=0.5)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-prob-synapse-diff-j-noise-curr.pdf")
plt.show()

# %% [markdown]
# Random curr

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.6),
        2: RandomCurrent(mean=6, std=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.6),
        2: RandomCurrent(mean=6, std=0.2),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.6),
        2: RandomCurrent(mean=6, std=0.2),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.25)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=25, p=0.1, variance=0.25)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=35, p=0.1, variance=0.25)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2, text_y=0.5)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2, text_y=0.5)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2, text_y=0.5)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-prob-synapse-diff-j-rand-curr.pdf")
plt.show()

# %% [markdown]
# ### Change in variance

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6),
        # 2: RandomCurrent(mean=6, std=0.2, init_I=6),
        3: NoisyCurrent(iterations=301, seed=1),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6),
        # 2: RandomCurrent(mean=6, std=0.2, init_I=6),
        3: NoisyCurrent(iterations=301, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6),
        # 2: RandomCurrent(mean=6, std=0.2, init_I=6),
        3: NoisyCurrent(iterations=301, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.1)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.5)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.9)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
axs[2, 0].text(0, 0.5, "same random current with mean 6", transform=axs[2, 0].transAxes, bbox=dict(facecolor='white', alpha=0.5))
ng_exc2.add_current_plot(axs[2, 1])
axs[2, 1].text(0, 0.5, "same random current with mean 6", transform=axs[2, 1].transAxes, bbox=dict(facecolor='white', alpha=0.5))
ng_exc3.add_current_plot(axs[2, 2])
axs[2, 2].text(0, 0.5, "same random current with mean 6", transform=axs[2, 2].transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-prob-synapse-diff-variance-same-rand-curr.pdf")
plt.show()

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.0),
        2: RandomCurrent(mean=6, std=0.2, init_I=6),
        # 3: NoisyCurrent(iterations=301, seed=1),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.0),
        2: RandomCurrent(mean=6, std=0.2, init_I=6),
        # 3: NoisyCurrent(iterations=301, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.0),
        2: RandomCurrent(mean=6, std=0.2, init_I=6),
        # 3: NoisyCurrent(iterations=301, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.1)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.5)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.9)})
sim.simulate(iterations=300)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2, text_y=0.5)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2, text_y=0.5)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2, text_y=0.5)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-prob-synapse-diff-variance-rand-curr.pdf")
plt.show()

# %% [markdown]
# ### Change in p

# %% [markdown]
# #### Same noisy Current

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6),
        # 2: RandomCurrent(mean=6, std=0.2, init_I=6),
        3: NoisyCurrent(iterations=501, seed=1),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6),
        # 2: RandomCurrent(mean=6, std=0.2, init_I=6),
        3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6),
        # 2: RandomCurrent(mean=6, std=0.2, init_I=6),
        3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.2)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.5, variance=0.2)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.9, variance=0.2)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
axs[2, 0].text(0, 0.5, "same noise current with mean 6", transform=axs[2, 0].transAxes, bbox=dict(facecolor='white', alpha=0.5))
ng_exc2.add_current_plot(axs[2, 1])
axs[2, 1].text(0, 0.5, "same noise current with mean 6", transform=axs[2, 1].transAxes, bbox=dict(facecolor='white', alpha=0.5))
ng_exc3.add_current_plot(axs[2, 2])
axs[2, 2].text(0, 0.5, "same noise current with mean 6", transform=axs[2, 2].transAxes, bbox=dict(facecolor='white', alpha=0.5))
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-prob-synapse-diff-p-same-noise-curr.pdf")
plt.show()

# %% [markdown]
# #### NOT Same Current

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.6),
        # 2: RandomCurrent(mean=6, std=0.2, init_I=6),
        # 3: NoisyCurrent(iterations=501, seed=1),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.6),
        # 2: RandomCurrent(mean=6, std=0.2, init_I=6),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.6),
        # 2: RandomCurrent(mean=6, std=0.2, init_I=6),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.2)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.5, variance=0.2)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.9, variance=0.2)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2, text_y=0.5)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2, text_y=0.5)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2, text_y=0.5)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-prob-synapse-diff-p-noise-curr.pdf")
plt.show()

# %% [markdown]
# #### Radnom current

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6),
        2: RandomCurrent(mean=6, std=0.2, init_I=6),
        # 3: NoisyCurrent(iterations=501, seed=1),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6),
        2: RandomCurrent(mean=6, std=0.2, init_I=6),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6),
        2: RandomCurrent(mean=6, std=0.2, init_I=6),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.2)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.5, variance=0.2)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.9, variance=0.2)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2, text_y=0.5)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2, text_y=0.5)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2, text_y=0.5)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-prob-synapse-diff-p-rand-curr.pdf")
plt.show()

# %% [markdown]
# ## Two population

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.0),
        # 2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.0),
        # 2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.25)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-prob-synapse-const-curr.pdf")
plt.show()

# %% [markdown]
# #### Random Current

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.25)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-prob-synapse-rand-curr.pdf")
plt.show()

# %% [markdown]
# ###  Change in j0

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.0),
        2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.0),
        2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=30, p=0.1, variance=0.25)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=30, p=0.1, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-prob-synapse-high-j.pdf")
plt.show()

# %% [markdown]
# High j0

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.0),
        2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.0),
        2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=50, p=0.1, variance=0.25)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-prob-synapse-rand-curr-diff-j.pdf")
plt.show()

# %% [markdown]
# ### Change in variance

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.0),
        2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.0),
        2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.75)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.75)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-prob-synapse-rand-curr-high-variance.pdf")
plt.show()

# %% [markdown]
# Different variance

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.0),
        2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.0),
        2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.75)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-prob-synapse-rand-curr-diff-variance.pdf")
plt.show()

# %% [markdown]
# ### Change in p

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.0),
        2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.0),
        2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.75, variance=0.25)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.75, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-prob-synapse-high-p-rand-curr.pdf")
plt.show()

# %% [markdown]
# Different p

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.0),
        2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.0),
        2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.7, variance=0.25)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedProbSynapse(j0=15, p=0.1, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-prob-synapse-diff-p-rand-curr.pdf")
plt.show()

# %% [markdown]
# # 3. Random coupling: Fixed number of presynaptic partners

# %% [markdown]
# ## One population

# %% [markdown]
# const curr

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc = sim.add_neuron_group(
    tag="ng_exc",
    size=250,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.0),
        # 2: RandomCurrent(mean=6, sts=0.2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc_event")
    }
)

syn = sim.add_synapse_group(
                   tag="exc-exc",
                   src=ng_exc,
                   dst=ng_exc,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=2, n=10, variance=0.25)})

sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(2, 2, figsize=(18, 8), sharex=True)
ng_exc.add_membrane_potential_plot(axs[0, 0], model_idx=7)
ng_exc.add_neuron_model_params_info(axs[0, 0], model_idx=7)

ng_exc.add_current_plot(axs[1, 0])
ng_exc.add_current_params_info(axs[1,0], current_idx=2)

ng_exc.add_raster_plot(axs[0, 1])
ng_exc.add_activity_plot(axs[1, 1])
fig.suptitle("Activity for one neuron group", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-fixed-synapse-const-curr.pdf")
plt.show()

# %% [markdown]
# random curr

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc = sim.add_neuron_group(
    tag="ng_exc",
    size=250,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.2),
        2: RandomCurrent(mean=6, std=0.1, init_I=6),
        # 3: NoisyCurrent(iterations=501, std=2),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc_event")
    }
)

syn = sim.add_synapse_group(
                   tag="exc-exc",
                   src=ng_exc,
                   dst=ng_exc,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=2, n=10, variance=0.25)})

sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(2, 2, figsize=(18, 8), sharex=True)
ng_exc.add_membrane_potential_plot(axs[0, 0], model_idx=7)
ng_exc.add_neuron_model_params_info(axs[0, 0], model_idx=7)

ng_exc.add_current_plot(axs[1, 0])
ng_exc.add_current_params_info(axs[1,0], current_idx=2)

ng_exc.add_raster_plot(axs[0, 1])
ng_exc.add_activity_plot(axs[1, 1])
fig.suptitle("Activity for one neuron group", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-fixed-synapse-rand-curr.pdf")
plt.show()

# %% [markdown]
# ### Change in j0

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.5),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.5),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=0.5),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=2, n=10, variance=0.25)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=5, n=10, variance=0.25)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=10, n=10, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-fixed-synapse-diff-j-noise-curr.pdf")
plt.show()

# %% [markdown]
# Rand curr

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.5),
        2: RandomCurrent(mean=6, std=0.1, init_I=6),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.5),
        2: RandomCurrent(mean=6, std=0.1, init_I=6),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.5),
        2: RandomCurrent(mean=6, std=0.1, init_I=6),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=2, n=10, variance=0.25)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=5, n=10, variance=0.25)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=10, n=10, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-fixed-synapse-diff-j-rand-curr.pdf")
plt.show()

# %% [markdown]
# ### Change in variance

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.5),
        2: RandomCurrent(mean=6, std=0.1, init_I=6),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.5),
        2: RandomCurrent(mean=6, std=0.1, init_I=6),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.5),
        2: RandomCurrent(mean=6, std=0.1, init_I=6),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=5, n=10, variance=0.1)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=5, n=10, variance=0.5)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=5, n=10, variance=0.9)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-fixed-synapse-diff-variance-rand-curr.pdf")
plt.show()

# %% [markdown]
# ### Change in n

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.5),
        2: RandomCurrent(mean=6, std=0.1, init_I=6),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.5),
        2: RandomCurrent(mean=6, std=0.1, init_I=6),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)
ng_exc3 = sim.add_neuron_group(
    tag="ng_exc3",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=0.5),
        2: RandomCurrent(mean=6, std=0.1, init_I=6),
        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc3_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc3_event")
    }
)
syn_exc1 = sim.add_synapse_group(
                   tag="exc1-exc1",
                   src=ng_exc1,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=7, n=10, variance=0.25)})
syn_exc2 = sim.add_synapse_group(
                   tag="exc2-exc2",
                   src=ng_exc2,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=7, n=25, variance=0.25)})
syn_exc3 = sim.add_synapse_group(
                   tag="exc3-exc3",
                   src=ng_exc3,
                   dst=ng_exc3,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=7, n=50, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot:
ng_exc1.add_raster_plot(axs[0, 0])
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
ng_exc2.add_raster_plot(axs[0, 1])
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)
ng_exc3.add_raster_plot(axs[0, 2])
ng_exc3.add_neuron_model_params_info(axs[0, 2], model_idx=7)

ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1.add_synapses_params_info(axs[1, 0], synapse_idx=3)
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2.add_synapses_params_info(axs[1, 1], synapse_idx=3)
ng_exc3.add_activity_plot(axs[1, 2])
syn_exc3.add_synapses_params_info(axs[1, 2], synapse_idx=3)

ng_exc1.add_current_plot(axs[2, 0])
ng_exc1.add_current_params_info(axs[2, 0], current_idx=2)
ng_exc2.add_current_plot(axs[2, 1])
ng_exc1.add_current_params_info(axs[2, 1], current_idx=2)
ng_exc3.add_current_plot(axs[2, 2])
ng_exc1.add_current_params_info(axs[2, 2], current_idx=2)
fig.suptitle("Activity for neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-one-ng-fixed-synapse-diff-n-rand-curr.pdf")
plt.show()

# %% [markdown]
# ## Two population

# %% [markdown]
# noise current

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=1),
        # 2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        2: ConstantCurrent(value=6, noise_range=1),
        # 2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=7, n=25, variance=0.25)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=7, n=25, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-fixed-synapse-noise-curr.pdf")
plt.show()

# %% [markdown]
# Random current

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=7, n=25, variance=0.25)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=7, n=25, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-fixed-synapse-rand-curr.pdf")
plt.show()

# %% [markdown]
# ### Change in j0

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=25, n=10, variance=0.25)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=25, n=10, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-fixed-synapse-high-j-rand-curr.pdf")
plt.show()

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=25, n=10, variance=0.25)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=7, n=10, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-fixed-synapse-diff-j-rand-curr.pdf")
plt.show()

# %% [markdown]
# ### Change in variance

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=15, n=10, variance=0.25)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=15, n=10, variance=0.9)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-fixed-synapse-diff-variance-rand-curr.pdf")
plt.show()

# %% [markdown]
# ### Change in n

# %% [markdown]
# #### same j

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=15, n=50, variance=0.25)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=15, n=50, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-fixed-synapse-high-n-same-j-rand-curr.pdf")
plt.show()

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=15, n=50, variance=0.25)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=15, n=10, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-fixed-synapse-same-j-diff-n.pdf")
plt.show()

# %%
sim = Simulation(net=Network(behavior={1: TimeResolution(dt=1.0,)}),)
ng_exc1 = sim.add_neuron_group(
    tag="ng_exc1",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc1_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc1_event")
    }
)
ng_exc2 = sim.add_neuron_group(
    tag="ng_exc2",
    size=100,
    behavior={
        1: SetCurrent(value=0),
        # 2: ConstantCurrent(value=6, noise_range=1),
        2: RandomCurrent(mean=6, sts=0.2, init_I="normal(6,1)"),
        # 3: NoisyCurrent(iterations=501, seed=1),

        6: Dendrite(),
        7: LIF(
            tau=10,
            u_rest=-65,
            u_reset=-70,
            threshold=-55,
            R=1.7,
            u_init="normal(-67,10)"
        ),
        8: ActivityRecorder(),
        9: Recorder(variables=["u", "I", "inp_I", "activity"], tag="ng_exc2_rec"),
        10: EventRecorder(variables=['spike'], tag="ng_exc2_event")
    }
)

syn_exc1_exc2 = sim.add_synapse_group(
                   tag="exc1-exc2",
                   src=ng_exc1,
                   dst=ng_exc2,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=50, n=50, variance=0.25)})

syn_exc2_exc1 = sim.add_synapse_group(
                   tag="exc2-exc1",
                   src=ng_exc2,
                   dst=ng_exc1,
                   behavior={3: RandomConnectedFixedInputSynapse(j0=10, n=10, variance=0.25)})
sim.simulate(iterations=500)

# %%
fig, axs = plt.subplots(3, 2, figsize=(18, 8), sharex=True, sharey='row')
# Raster plot for ng1:
ng_exc1.add_raster_plot(axs[0, 0], s=1)
ng_exc1.add_neuron_model_params_info(axs[0, 0], model_idx=7)
# Raster plot for ng2:
ng_exc2.add_raster_plot(axs[0, 1], s=1)
ng_exc2.add_neuron_model_params_info(axs[0, 1], model_idx=7)

# Activity plot for ng1:
ng_exc1.add_activity_plot(axs[1, 0])
syn_exc1_exc2.add_synapses_params_info(axs[1, 0], synapse_idx=3)
# Activity plot for ng2:
ng_exc2.add_activity_plot(axs[1, 1])
syn_exc2_exc1.add_synapses_params_info(axs[1, 1], synapse_idx=3)

# Current plot for ng1:
ng_exc1.add_current_plot(axs[2, 0])
ng_exc2.add_current_params_info(axs[2, 0], current_idx=2)
# Current plot for ng2:
ng_exc2.add_current_plot(axs[2, 1])
ng_exc2.add_current_params_info(axs[2, 1], current_idx=2)


fig.suptitle("Activity of two excitatory neuron groups", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("report/plots/part2-two-ng-fixed-synapse-diff-n-rand-curr.pdf")
plt.show()


