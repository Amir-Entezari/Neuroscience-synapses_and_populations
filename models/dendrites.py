from pymonntorch import *


class Dendrite(Behavior):
    def forward(self, ng):
        for synapse in ng.afferent_synapses["All"]:
            ng.I += synapse.I
