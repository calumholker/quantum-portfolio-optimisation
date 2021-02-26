import pennylane as qml
import numpy as np
import tensorflow as tf

wires = range(32)
device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
my_bucket = "amazon-braket-080834" # the name of the bucket
my_prefix = "QGAN_1" # the name of the folder in the bucket
s3_folder = (my_bucket, my_prefix)
dev = qml.device(
    "braket.aws.qubit",
    device_arn=device_arn,
    wires=wires,
    s3_destination_folder=s3_folder,
    parallel=True,
)

@qml.qnode(dev)
def gen_circuit(gen_weights):
    qml.broadcast(unitary=qml.RY, pattern = 'single', wires = wires, parameters = gen_weights)
    return [qml.expval(qml.PauliX(i)) for i in range(8)]

def gen_cost(gen_weights):
    return -np.array(gen_circuit(gen_weights))

def train_generator(gen_weights):
    opt = qml.AdamOptimizer(0.1)
    cost = lambda: gen_cost(gen_weights)
    for _ in range(50):
        gen_weights = opt.step(lambda gen_weights: gen_cost(gen_weights), gen_weights)

eps = 1e-2
k = 3
gen_weights = np.random.normal(scale=eps, size=(32))
# gen_weights = tf.Variable(init_gen_weights)

train_generator(gen_weights)

# np.save('gen_weights', gen_weights)
# model.save('disc')