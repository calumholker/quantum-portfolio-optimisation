from tensorflow.keras import callbacks
import remote_cirq
import pennylane as qml
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import TensorBoard

qml.enable_tape()
wires = range(32)
API_KEY = "AIzaSyAkLyvGHAIGGm8kT5SbzJB0Wi7dCT_4kPQ"
device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
sim = remote_cirq.RemoteSimulator(API_KEY)
my_bucket = "amazon-braket-080834" # the name of the bucket
my_prefix = "QGAN_1" # the name of the folder in the bucket
s3_folder = (my_bucket, my_prefix)

# Defines FLOQ Device
dev = qml.device("cirq.simulator",
                 wires=32,
                 simulator=sim,
                 analytic=False)

# Defines Braket SV1 Device
dev_remote = qml.device(
    "braket.aws.qubit",
    device_arn=device_arn,
    wires=wires,
    s3_destination_folder=s3_folder,
    parallel=True,
)

data = np.load('data/preprocessed/prices_32_8.npy', allow_pickle=True) # Import preprocessed data

# Adjust format of data
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

#Create Discriminator Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu', input_shape=[1, 40]))
model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=1, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Defines Generator Model
def generator(w):
    qml.broadcast(unitary=qml.RY, pattern = 'single', wires = wires, parameters = w[0:32])
    for k in range(1, 4):
        qml.broadcast(unitary=qml.RY, pattern = 'single', wires = wires, parameters = w[(32*k):(32*(k+1))])
        qml.broadcast(unitary=qml.CZ, pattern = 'ring', wires=wires)

@qml.qnode(dev)
def gen_circuit(array, gen_weights):
    qml.templates.AngleEmbedding(array, wires, rotation='Y')
    generator(gen_weights)
    return [qml.expval(qml.PauliX(i)) for i in range(8)]

def prob_fake_true(array, gen_weights):
    w = array[:32]
    try:
        f = (gen_circuit(w, gen_weights))._value
    except:
        f = np.array((gen_circuit(w, gen_weights)))
    for i in f:
        w = np.append(w, i)
    tensor = tf.convert_to_tensor(np.array(w).reshape(1,1,40))
    fake_disc_output = float(model(tensor)[0][0])
    prob_fake_true = (fake_disc_output + 1) / 2
    return prob_fake_true

def gen_cost(gen_weights):
    return -prob_fake_true(dat, gen_weights)

def train_discriminator(model, data, y):
    train_x = data
    train_y = y

    """Code for creating training data set, uncomment if doing more than one epoch"""
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

    """Imports already created training data sets"""
    data = np.load('train_x.npy', allow_pickle=True)
    data = np.load('train_y.npy', allow_pickle=True)

    train_x, train_y = shuffle(train_x, train_y)

    NAME = 'Discriminator_' 
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=50, validation_split=0.3, callbacks=[tensorboard])
    model.save('disc')

def train_generator(gen_weights):
    opt = qml.AdamOptimizer(0.1)
    costs = np.array([])
    for _ in tqdm(range(1)): # Number of epochs
        for i in tqdm(range(len(data))):
            global dat
            dat = data[i][0]
            gen_weights, cost = opt.step_and_cost(lambda gen_weights: gen_cost(gen_weights), gen_weights)
            np.append(costs, cost)
            np.save(f'gen-weights/gen_weights_{_}_{i}', gen_weights)
            np.save(f'costs/cost_{_}_{i}', costs)
        
"""Required to set initial weights for data, example commented below"""
# eps = 1e-2
# k = 3
# init_gen_weights = np.array([np.pi] + [0] * (32*(k+1)-1)) + np.random.normal(scale=eps, size=(32*(k+1),))
gen_weights = tf.Variable(init_gen_weights)

# Can be looped for more iterations
train_discriminator(model, data, y)
train_generator(gen_weights)

np.save('gen_weights', gen_weights)
model.save('disc')
