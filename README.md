# Using Quantum Generative Adversarial Networks for Portfolio Optimisation
Calum Holker, Pavan Jayasinha, Aarsh Chaube, Tomasz Andrzejewski <br>
Submission for QHack open hackathon for team QLords <br><br>

Blog Post: [Using Quantum Generative Adversarial Networks for Portfolio Optimisation](https://calumholker.medium.com/using-quantum-generative-adversarial-networks-for-portfolio-analysis-f8c56ac68fd2)

## Project Overview
The essence of our method is that we will end up with a model that takes in a set of data defining the stock prices of several stocks over a defined number of previous days. The model will then output it's predicted stock data for the next period of days. We train this using a second network, a discriminator whose goal is to determine if the data (both the previous time period and the following time period) is generated or real. The generator is then trained so that the discriminator can not tell the difference between the real and fake data. These two networks are trained in turn with large quantities of data. This results in a generator that produces data that the discriminator cannot decipher if it has been generated or real. In the following diagram the training data M is split into the previous data Mb and the future data Mf. Mb can then be input into the generator, and with some noise latent vector a generated sample Mf^ can be produced. These two datasets M (Mb+Mf) and M' (Mb+Mf^) can then be used in the training methods. <br>
![alt text](https://github.com/calumholker/quantum-portfolio-optimisation/blob/master/qGAN/images/gans.png "GAN")
Due to restrictions such as time and current quantum computer capability, we apply this method only for one epoch on 100 sequences of data. Even from this we see that the model has improved, and it is easy to see that if this was extrapolated, the resulting model would be an effective one. <br>
We further look into the use of QAOA and VQE for analysing this data if we had the resources to fully produce it.

## Built With
PennyLane <br>
Qiskit <br>
Amazon Braket <br>
Google FLOQ <br>

## File Structure
data/ - contains stock market data in sequences, and notbooks creating that data
qGAN/ - contains the implementation of qGANs on stock market data
QAOA/ - contains the implementation of QAOA for portfolio optimisation
testing-versions/ - 

## References
[(1) PAGAN: Portfolio Analysis with Generative Adversarial Networks](https://arxiv.org/pdf/1909.10578.pdf)
[(2) Quantum Generative Adversarial Networks for learning and loading random distributions](https://www.nature.com/articles/s41534-019-0223-2.pdf)
[(3) Enhancing Combinatorial Optimization with Quantum Generative Models](https://www.zapatacomputing.com/wp-content/uploads/2020/12/2101.06250.pdf)
[(4) Improving Variational Quantum Optimization using CVaR] (https://arxiv.org/pdf/1907.04769.pdf)
[(5) Qiskit Aqua Portfolio Optimisation](https://qiskit.org/documentation/tutorials/finance/01_portfolio_optimization.html)
