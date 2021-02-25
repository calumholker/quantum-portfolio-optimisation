import sys
import remote_cirq
import pennylane as qml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle

qml.enable_tape()
wires = range(32)
API_KEY = "AIzaSyAkLyvGHAIGGm8kT5SbzJB0Wi7dCT_4kPQ"
device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
sim = remote_cirq.RemoteSimulator(API_KEY)
my_bucket = "amazon-braket-080834" # the name of the bucket
my_prefix = "QGAN_1" # the name of the folder in the bucket
s3_folder = (my_bucket, my_prefix)
dev = qml.device("cirq.simulator",
                 wires=32,
                 simulator=sim,
                 analytic=False)
dev_remote = qml.device(
    "braket.aws.qubit",
    device_arn=device_arn,
    wires=wires,
    s3_destination_folder=s3_folder,
    parallel=True,
)
data = np.load('data/preprocessed/prices_32_8.npy', allow_pickle=True)

train = []
for a in range(100):
    for b in range(1):
        arr1 = [[]]
        arr1[0] = list(data[b][a][0])
        for bit in data[b][a][1]:
            arr1[0].append(bit)
        train.append(arr1)
data = np.array(train)
y = np.ones(len(data))

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

def prob_fake_true(array, gen_weights):
    w = array[:32]
    f = np.array(gen_circuit(w, gen_weights))
    for i in f:
        w = np.append(w, i)
    tensor = tf.convert_to_tensor(np.array(w).reshape(1,1,40))
    fake_disc_output = model(tensor)[0][0]
    prob_fake_true = (fake_disc_output + 1) / 2
    return prob_fake_true

def gen_cost(gen_weights):
    return -prob_fake_true(dat, gen_weights)

def train_discriminator(model, data, y):
    train_x = data
    train_y = y

    # for i in tqdm(range(len(data))):
    #     set = data[i][0]
    #     w = set[:32]
    #     f = np.array(gen_circuit(w, gen_weights))
    #     for i in f:
    #         w = np.append(w,i)
    #     np.append(train_x, [w])
    #     np.append(train_y, 0)

    # np.save('train_x', train_x)
    # np.save('train_y', train_y)

    data = np.load('train_x.npy', allow_pickle=True)
    data = np.load('train_y.npy', allow_pickle=True)

    train_x, train_y = shuffle(train_x, train_y)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=50, validation_split=0.3)

def train_generator():
    opt = tf.keras.optimizers.SGD(0.1)
    cost = lambda: gen_cost(gen_weights)
    for batch in data:
            global dat
            dat = batch[0]
            for step in range(20):
                print(step)
                opt.minimize(cost, gen_weights)
                cost_val = cost().numpy()
                print("Step {}: cost = {}".format(step, cost_val))

eps = 1e-2
k = 3
init_gen_weights = np.array([np.pi] + [0] * (32*(k+1)-1)) + np.random.normal(scale=eps, size=(32*(k+1),))
gen_weights = tf.Variable(init_gen_weights)

train_discriminator(model, data, y)
train_generator()

np.save('models/generator-weights/gen_weights', gen_weights)
model.save('models/discriminator/disc')