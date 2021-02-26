import remote_cirq
import pennylane as qml
import numpy as np
from braket.aws import AwsDevice
wires = 26
np.random.seed(0)

qml.enable_tape()
weights = np.random.randn(1, wires, 3)
API_KEY = "AIzaSyAkLyvGHAIGGm8kT5SbzJB0Wi7dCT_4kPQ"
device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
sim = remote_cirq.RemoteSimulator(API_KEY)
my_bucket = "amazon-braket-080834" # the name of the bucket
my_prefix = "QGAN_1" # the name of the folder in the bucket
s3_folder = (my_bucket, my_prefix)

dev = qml.device("cirq.simulator",
                 wires=wires,
                 simulator=sim,
                 analytic=False)

dev_remote = qml.device(
    "braket.aws.qubit",
    device_arn=device_arn,
    wires=wires,
    s3_destination_folder=s3_folder,
    parallel=True,
)

@qml.qnode(dev)
def my_circuit(weights):
	qml.templates.layers.StronglyEntanglingLayers(weights,
	                                              wires=range(wires))
	return qml.expval(qml.PauliZ(0))

print(my_circuit(weights))