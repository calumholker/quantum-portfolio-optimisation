## Applying qGANs to Stock Data
Calum Holker, Pavan Jayasinha, Aarsh Chaube, Tomasz Andrzejewski <br>
Submission for QHack open hackathon for team QLords <br><br>

Blog Post: [Using Quantum Generative Adversarial Networks for Portfolio Optimisation](https://calumholker.medium.com/using-quantum-generative-adversarial-networks-for-portfolio-analysis-f8c56ac68fd2)

# Method
The essence of our method is that we will end up with a model that takes in a set of data defining the stock prices of several stocks over a defined number of previous days. The model will then output it's predicted stock data for the next period of days. We train this using a second network, a discriminator whose goal is to determine if the data (both the previous time period and the following time period) is generated or real. The generator is then trained so that the discriminator can not tell the difference between the real and fake data. These two networks are trained in turn with large quantities of data. This results in a generator that produces data that the discriminator cannot decipher if it has been generated or real. In the following diagram the training data M is split into the previous data Mb and the future data Mf. Mb can then be input into the generator, and with some noise latent vector a generated sample Mf^ can be produced. These two datasets M (Mb+Mf) and M' (Mb+Mf^) can then be used in the training methods. <br><br>
![alt text](https://github.com/calumholker/quantum-portfolio-optimisation/blob/master/qGAN/images/gans.png "GAN") <br><br>

# Models
Ideally both the discriminator and the generator would be quantum. However the limitations on the number of qubits (32) means that compressing the data such that these two would work alongside each other would cause main features to be lost (this was tested). In our example implementation we have therefore used a classical discriminator and a quantum generator, allowing all qubits to be used as input for the generator. For the discriminator we have used a convolutional neural network as this has been shown to handle time series data well. For the generator we have used the quantum implementation below, taken from [2]. <br><br>
![alt text](https://github.com/calumholker/quantum-portfolio-optimisation/blob/master/qGAN/images/generator.png "generator") <br><br>
We have implemented this circuit with k=3, and 32 qubits, meaning there are 128 trainable parameters. We used the amplitude embedding method from PennyLane to encode the data into the intial state. For the output we take the expectation of the first 8 qubits. This means that our input data is a sequence of 32 days and it outputs the sequence for the following 8 days. Some validation data was also created in order to not overfit the model.

# Training
Due to time limitations, we were only able to run one and a half epochs for training the generator (where each epoch cycles through and trains the generator on each of the 100 sequences), and only train the discriminator once prior. <br><br>
First training the classical discriminator on data produced by an initial generator that had not been trained at all. As expected this was very effective an quickly increased accuracy tending towards 100%, as shown in the tensorboard graphs below.<br><br>
![alt text](https://github.com/calumholker/quantum-portfolio-optimisation/blob/master/qGAN/images/training-loss.png "tensorboard") <br><br>
Training the generator was less successful, due to the limited data and epochs input. However after one epoch the generated data was significantly closer to the real validation data. In 54% of cases the model correctly predicted if the stock would increase or decrease in the next period, compared to 49% without training. This is a good accuracy for one epoch for stock prediction, in general an accuracy of 60% is widely accepted to be a good model. <br><br>
![alt text](https://github.com/calumholker/quantum-portfolio-optimisation/blob/master/qGAN/results/Data.png "result") <br><br>

# Devices
To run these models we used an array of simulators and devices: <br><br>
FLOQ - Having won access to Google's in development quantum simulator it became incredibly useful in our training process. The simulator significantly sped up the process of creating the code and debugging it so that time on the real devices was best spent. FLOQ is optimised for 32 qubits which is ideal for our circuit, and sped up the circuit run time from O(500 seconds) on SV1 to O(50 seconds). This allowed our preliminary testing to be as quick as possible. <br>
SV1 - Amazon Braket's SV1 simulator proved useful in testing the optimisation of the circuit parameters, as the parallel feature meant that multiple circuits could run in parallel, this was used to finalise our testing of the model. <br>
Rigetti - Having won the $4000 AWS credits power up, we could run our final model through Amazon Braket on the Rigetti machine. <br>

# Limitations
For this project we had the following limitations imposed upon us: <br><br>
1. The devices available imposed a maximum of 32 qubits. <br>
2. The run time of the training could not be longer than the period Rigetti was open for. <br><br>

This meant a few compromises were made, that would be changed given bigger computers or more time. The following method would be used in the ideal case:<br><br>
All of the 22350 sequence datasets produced would be used in the training of the model, as modelling time series data takes a notoriously large amount of data to become effective. Furthermore each sequence could be expanded to include more data and other indicators such as volume. Then, when implementing the models, the generator would take in the data for all assets at once and output the predictions for all of the assets. If the datasets are larger a quantum CNN could also be implemented on the large quantity of data, as it is in the classical implementation of GANs for time series as described in [1]. Large datasets would be needed for this in order to not lose any key information in the convolutions. Finally the discriminator would also be quantum and can be connected to the same output wires of the generator, meaning that the GAN is entirely quantum rather than partially. <br><br>
If this model was implemented with more iterations of training the generator and discriminator alternately it would extrapolate the small increase we saw in our implementation and produce a much better model that can then be used for portfolio analysis.

# Using the Model for Portfolio Analysis
As the data our model produced is not complete, for the portfolio analysis implementation using QAOA we take raw historical data to compare the quantum methods against classical benchmarks. If we had the model described above the data used in this would be replaced with the predictions made by the generator, allowing for future prices to be accounted for in the portfolio model. Furthermore the accuracy of the predictions can be included and a combination of past data and future predictions, weighted according to this risk of the predictions being incorrect can be incorporated into solving the portfolio optimisation problem.
