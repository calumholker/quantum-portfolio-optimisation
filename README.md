# Using Quantum Generative Adversarial Networks for Portfolio Optimisation
Calum Holker, Pavan Jayasinha, Aarsh Chaube, Tomasz Andrzejewski <br>
Submission for QHack open hackathon for team QLords <br><br>

Blog Post: [Using Quantum Generative Adversarial Networks for Portfolio Optimisation](https://calumholker.medium.com/using-quantum-generative-adversarial-networks-for-portfolio-analysis-f8c56ac68fd2)

## Project Summary
We use qGANs on stock market data to create a model that can predict future trends, and look at how that data can then be used to optimise portfolios.

## Built With
PennyLane <br>
Qiskit <br>
Amazon Braket <br>
Google FLOQ <br>

## File Structure
data/ - contains stock market data in sequences, and notbooks creating that data <br>
qGAN/ - contains the implementation of qGANs on stock market data <br>
QAOA-VQE/ - contains the implementation of QAOA and VQE for portfolio optimisation <br>
testing-versions/ - contains files used for testing the main files

## References
[(1) PAGAN: Portfolio Analysis with Generative Adversarial Networks](https://arxiv.org/pdf/1909.10578.pdf) <br>
[(2) Quantum Generative Adversarial Networks for learning and loading random distributions](https://www.nature.com/articles/s41534-019-0223-2.pdf) <br>
[(3) Enhancing Combinatorial Optimization with Quantum Generative Models](https://www.zapatacomputing.com/wp-content/uploads/2020/12/2101.06250.pdf) <br>
[(4) Improving Variational Quantum Optimization using CVaR] (https://arxiv.org/pdf/1907.04769.pdf) <br>
[(5) Qiskit Aqua Portfolio Optimisation](https://qiskit.org/documentation/tutorials/finance/01_portfolio_optimization.html) <br>
