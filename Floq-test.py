import remote_cirq
import pennylane as qml
import numpy as np
import sys
wires = 26
np.random.seed(0)

weights = np.random.randn(1, wires, 3)
API_KEY = "AIzaSyAkLyvGHAIGGm8kT5SbzJB0Wi7dCT_4kPQ"
sim = remote_cirq.RemoteSimulator(API_KEY)

dev = qml.device("cirq.simulator",
                 wires=wires,
                 simulator=sim,
                 analytic=False)

@qml.qnode(dev)
def my_circuit(weights):
	qml.templates.layers.StronglyEntanglingLayers(weights,
	                                              wires=range(wires))
	return qml.expval(qml.PauliZ(0))

print(my_circuit(weights))