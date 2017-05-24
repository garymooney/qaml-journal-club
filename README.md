Quantum Algorithms and Machine Learning Reading Group
======
This is a reading material archive for a journal club held at Melbourne University. 

A paper will be sent out every fortnight. The goal is to openly discuss it at various depths as a group. So no matter what your background is, we will all hopefully learn something. The meetings are casual, so there's no pressure to present or anything like that.

In the past weeks we have covered the following papers:

__1. Quantum Algorithm for Linear Systems of Equations (HHL)__ - Harrow, Hassidim and Lloyd.  
  * University of Bristol and MIT.

> Harrow, Aram W., Avinatan Hassidim, and Seth Lloyd. [Physical review letters](https://doi.org/10.1103/PhysRevLett.103.150502) 103.15 (7-Oct-09): 150502.

> Harrow, Aram W., Avinatan Hassidim, and Seth Lloyd. arXiv preprint [arXiv:0811.3171v3](https://arxiv.org/abs/0811.3171) (30-Sep-09).

As the title suggests this paper introduces a new quantum algorithm for solving linear systems of equations. Classically, the best time scaling algorithm is roughly N sqrt(κ) where κ is something called the condition number (how sensitive the matrix is to changes). This new quantum algorithm scales roughly poly log(N) and poly (κ), which is essentially exponentially faster. There is a catch though, the actual readout of the solution scales linearly (which is the same as classical). So the algorithm will be mainly beneficial for feeding the output into the input of some other quantum algorithm or approximating the expectation value.

The method uses a technique where a non-unitary transformation involving measurement is used within the quantum circuit. Computation is only continued if a particular qubit state is measured, otherwise it is repeated.

We later found that this idea could be extended to [linear differential equations](https://arxiv.org/abs/1010.2745)
and [non-linear differential equations](https://arxiv.org/abs/0812.4423)

__2. Quantum Algorithms for Supervised and Unsupervised Machine Learning__ - Lloyd, Mohseni and Rebentrost.  
  * MIT and Google.

> Lloyd, Seth, Masoud Mohseni, and Patrick Rebentrost. arXiv preprint [arXiv:1307.0411](https://arxiv.org/abs/1307.0411) (4-Nov-13).

This paper focuses on applying quantum algorithms on supervised and unsupervised clustering algorithms. 

The supervised algorithm (classification) is simply the process of assigning each new data point to a class with the closest mean. The  distances to the means are composed of two factors. The paper describes a method to find one factor by applying a procedure called a swap test between a particular qubit register state representing the mean and an ancilla qudit (not qubit) representing the new datum. The second factor can be found using Grover’s algorithm or quantum counting. This algorithm can, in theory, obtain exponential speedup over the classical version.

The unsupervised algorithm (clustering) is for k-means clustering which uses Lloyd’s algorithm (not related to the author of this paper). There are a few visualisations of it around the net (It’s a pretty cool algorithm, so you should check it out). The paper introduces the quantum Lloyd’s algorithm with speed up (although not exponential) over the classical version. Unfortunately, we didn’t get enough time to go through this section in detail.

I think the take home message of this paper was that quantum computers are powerful when it comes to manipulating large numbers of high-dimensional vector spaces, which is precisely what is required for vector-based machine learning. This paper runs through some examples of how existing quantum algorithms can be directly applied to supervised and unsupervised machine learning clustering problems.

__3. Quantum Sampling Problems, BosonSampling and Quantum Supremacy__ – Lund, Bremner and Ralph.  
  * CQC2T.
  
> Lund, A. P., Michael J. Bremner, and T. C. Ralph. [npj Quantum Information](http://dx.doi.org/10.1038/s41534-017-0018-2) (13-Apr-17)

> Lund, A. P., Michael J. Bremner, and T. C. Ralph. arXiv preprint [arXiv:1702.03061](https://arxiv.org/abs/1307.0411) (10-Feb-17).

This paper is a review on quantum speedup for quantum sampling problems such as BosonSampling and IQP. IQP is essentially a generalisation of BosonSampling for commuting quantum gates on qubits. The paper reduces the argument of quantum supremacy for BosonSampling and IQP to be that either the polynomial hierarchy collapses to the third level or quantum algorithms are more powerful than classical. This is similar to the statement P = NP, except relative to an oracle. They go on to say that 50 photons (or qubits with sufficiently high fidelity for IQP) would be sufficient to demonstrate quantum supremacy. So quantum sampling is a promising direction that brings us much closer to demonstrating quantum supremacy.

__4. Application of Quantum Annealing to Training of Deep Neural Networks__ - Adachi and Henderson.  
  * Lockheed Martin Information Systems & Global Solutions.
  
> Adachi, Steven H., and Maxwell P. Henderson. arXiv preprint [arXiv:1510.06356](https://arxiv.org/abs/1510.06356) (21-Oct-15).

__5. Quantum Machine Learning over Infinite Dimensions__ - Lau, Pooser, Siopsis and Weedbrook.
  * Ulm University, Oak Ridge National Laboratory and The University of Tennessee.
  
> Lau, Hoi-Kwan, Raphael Pooser, George Siopsis, and Christian Weedbrook. [Physical Review Letters](https://doi.org/10.1103/PhysRevLett.118.080501) 118.8 (21-Feb-17): 080501.

> Lau, Hoi-Kwan, Raphael Pooser, George Siopsis, and Christian Weedbrook. arXiv preprint [arXiv:1603.06222v2](https://arxiv.org/abs/1603.06222) (14-Nov-16).

__6. Quantum machine learning with small-scale devices: Implementing a distance-based classifier with a quantum interference circuit__ - Schuld, Fingerhuth, and Petruccione.  
  * University of KwaZulu-Natal,  University of Maastricht and National Institute for Theoretical Physics (KwaZulu-Natal).

> Schuld, Maria, Mark Fingerhuth, and Francesco Petruccione. arXiv preprint [arXiv:1703.10793](https://arxiv.org/abs/1703.10793) (31-Mar-17).

__7. Quantum Random Access Memory__ - Giovannetti, Lloyd and Maccone.  
  * Scuola Normale Superiore, MIT and University of Pavia.
  
> Giovannetti, Vittorio, Seth Lloyd, and Lorenzo Maccone. [Physical review letters](https://doi.org/10.1103/PhysRevLett.100.160501) 100.16 (21-Apr-08): 160501.

> Giovannetti, Vittorio, Seth Lloyd, and Lorenzo Maccone. arXiv preprint [arXiv:0708.1879v2](https://arxiv.org/abs/0708.1879) (26-Mar-08).

Another paper by the same authors on qRAM that goes into more detail:

> Giovannetti, Vittorio, Seth Lloyd, and Lorenzo Maccone. "Architectures for a quantum random access memory." [Physical Review A](https://doi.org/10.1103/PhysRevA.78.052310) 78.5 (5-Nov-08): 052310.

> Giovannetti, Vittorio, Seth Lloyd, and Lorenzo Maccone. "Architectures for a quantum random access memory." arXiv preprint [arXiv:0807.4994v2](https://arxiv.org/abs/0807.4994) (11-Nov-08).

__8. Quantum gradient descent for linear systems and least squares__ - Kerenidis and Prakash  
  * Paris Diderot University and Nanyang Technological University

> Kerenidis, Iordanis, and Anupam Prakash. arXiv preprint [arXiv:1704.04992](https://arxiv.org/abs/1704.04992) (1-May-17).

Although difficult and long, this is a great paper full of the latest in quantum algorithmic techniques. There are two new major techniques that are introduced. 

The first is a generalized quantum Singular Value Estimation (SVE) procedure that is more computationally efficient than previous methods. The HHL algorithm has complexity O(s(A)<sup>2</sup>κ(A)<sup>2</sup> / ε) where s is the sparsity, κ is the condition number, ε is the algorithm error and the dependency of the dimension of matrix A has been supressed. Previous work has improved this algorithm to O(s(A)κ(A)log(1/ε)). By using this improved SVE procedure, the authors introduce a new parameter μ(A) such that μ(A) < s(A) and for sparse matrices μ(A) < sqrt[n] while s(A) has Ω(n) introducing polynomial speedup. This leads to an even faster algorithm with complexity O(μ(A)κ(A)log(1/ε)). Although this algorithm still doesn't have exponential savings on all matrices, it does have exponential savings on more matrices than previous methods. 

Current work on quantum gradient descent algorithms have a major issue where each iteration requires measurement of an ancilla qubit to be in a particular state to continue, if the measurement was not in this state the entire algorithm would have to start over. This led to an exponential suppression in propability of measuring the correct output state of the algorithm with respect to the number of iterations. The second major technique that this paper introduces overcomes this challenge. The final set of parameters can be written as the initial parameter plus a summation over all update changes (gradients multiplied by step size), for the gradient descent with affine updates, the gradients are linear. The idea is to use the improved SVE for matrix multiplication and to compute the summation of gradients in quantum parallel, which overcomes this exponential suppression. Amplitude amplification is used to increase the probability of selecting the required ancilla states. In many cases the overall expected running time for general quantum gradient descent turns out to be O(τ(C<sub>U</sub>log(τ))), where C<sub>U</sub> is the runtime cost for determining how much change an update has for a particular iteration step and τ is the number of iterations.

Most of the rest of the paper seems to go deep into the details and applications of the above while providing proofs for many of the methods. They define quantum gradient descent and go into detail about the correctness and running time. They show how their improved SVE procedure can be used to directly solve linear systems. Then they apply it to the linear update of quantum gradient descent, which can then be used to indirectly solve linear equations. These two methods are then compared. Finally they show how quantum gradient descent and stochastic quantum gradient descent can be used to solve the weighted least squares problem.

__9. Simulating a perceptron on a quantum computer__ - Schuld, Sinayskiy and Petruccione  
  * University of KwaZulu-Natal, and National Institute for Theoretical Physics (KwaZulu-Natal).

> Schuld, Maria, Ilya Sinayskiy, and Francesco Petruccione. [Physics Letters A](https://doi.org/10.1016/j.physleta.2014.11.061) 379.7 (20-Mar-15): 660-663.

> Schuld, Maria, Ilya Sinayskiy, and Francesco Petruccione. arXiv preprint [arXiv:1412.3635](https://arxiv.org/abs/1412.3635) (11-Dec-14).

__This Week: Quantum Machine Learning__ - Biamonte, Wittek, Pancotti, Rebentrost, Wiebe and Lloyd
  * University of Malta, University of Waterloo, ICFO - The Institute of Photonic Sciences (Spain), University of Borås (Sweden), Max Planck Insitute of Quantum Optics (Germany), MIT, Station Q (Microsoft Research).

> Biamonte, Jacob, et al. "Quantum Machine Learning." arXiv preprint [arXiv:1611.09347](https://arxiv.org/abs/1611.09347) (28-Nov-16).

## Interesting Papers (not yet covered)
[A Quantum Linear System Algorithm for Dense Matrices](https://arxiv.org/abs/1704.06174) (3-May-17) - Wossnig, Zhao and Prakash

[Deep Learning and Quantum Entanglement: Fundamental Connections with Implications to Network Design](https://arxiv.org/abs/1704.01552) (10-Apr-17) - Levine, Yakira, Cohen and Shashua

[A Survey of Quantum Learning Theory](https://arxiv.org/abs/1701.06806) (24-Jan-17) - Arunachalam and Wolf.

[Quantum Deep Learning](https://arxiv.org/abs/1412.3489) (22-May-15) - Wiebe, Kapoor and Svore

[An Introduction to Quantum Machine Learning](http://dx.doi.org/10.1080/00107514.2014.964942) (15-Oct-14) - Schuld, Sinayskiy, and Petruccione

## Useful Links
[colah's blog](http://colah.github.io/) for various machine learning concepts and in particular the post [Neural Networks, Manifolds, and Topology](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)

[TensorFlow Playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.54871&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false) lets you play with neural networks that are completely visualised. Very useful for gaining intuition for how they learn.
