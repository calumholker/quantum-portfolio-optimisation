import sys
import remote_cirq
import pennylane as qml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

wires = range(32)
API_KEY = "AIzaSyAkLyvGHAIGGm8kT5SbzJB0Wi7dCT_4kPQ"
sim = remote_cirq.RemoteSimulator(API_KEY)
dev = qml.device("cirq.simulator",
                 wires=32,
                 simulator=sim,
                 analytic=False)
data = np.load('data/preprocessed/prices_32_8.npy', allow_pickle=True)
train = []
for a in range(len(data[0])):
    for b in range(len(data)):
        arr1 = [[]]
        arr1[0] = list(data[b][a][0])
        for bit in data[b][a][1]:
            arr1[0].append(bit)
        train.append(arr1)
data = np.array(train)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu', input_shape=[1, 40]))
model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

def generator(w):
    qml.broadcast(unitary=qml.RY, pattern = 'single', wires = wires, parameters = w[0:32])
    for k in range(1, 4):
        qml.broadcast(unitary=qml.RY, pattern = 'single', wires = wires, parameters = w[(32*k):(32*(k+1))])
        qml.broadcast(unitary=qml.CZ, pattern = 'ring', wires=wires)

@qml.qnode(dev, interface="tf")
def gen_circuit(array, gen_weights):
    qml.templates.AngleEmbedding(array, wires, rotation='Y')
    generator(gen_weights)
    return [qml.expval(qml.PauliX(i)) for i in range(8)]


def prob_real_true(array):
    tensor = tf.convert_to_tensor(np.array(array).reshape(1,1,40))
    true_disc_output = model(tensor)[0][0]
    prob_real_true = (true_disc_output + 1) / 2
    return prob_real_true

def prob_fake_true(array, gen_weights):
    w = array[:32]
    f = np.array(gen_circuit(w, gen_weights))
    for i in f:
        w = np.append(w, i)
    tensor = tf.convert_to_tensor(np.array(w).reshape(1,1,40))
    fake_disc_output = model(tensor)[0][0]
    prob_fake_true = (fake_disc_output + 1) / 2
    return prob_fake_true

def disc_cost(disc_weights):
    w = []
    for i in range(len(disc_weights)):
        w.append(disc_weights[i].numpy())
    model.set_weights(w)
    cost = prob_fake_true(dat, gen_weights) - prob_real_true(dat)
    return cost

def gen_cost(array, gen_weights):
    return -prob_fake_true(array, gen_weights)

np.random.seed(0)
eps = 1e-2
k = 3
init_gen_weights = np.array([np.pi] + [0] * (32*(k+1)-1)) + np.random.normal(scale=eps, size=(32*(k+1),))
gen_weights = tf.Variable(init_gen_weights)
init_disc_weights = model.trainable_weights
disc_weights = []
for i in range(len(init_disc_weights)):
    disc_weights.append(tf.Variable(tf.convert_to_tensor(init_disc_weights[i])))


opt = tf.keras.optimizers.SGD(0.1)

cost = lambda: disc_cost(disc_weights)

for batch in data:
        dat = batch[0]
        for step in range(20):
            print(step)
            opt.minimize(cost, disc_weights)
            cost_val = cost().numpy()
            print("Step {}: cost = {}".format(step, cost_val))