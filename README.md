Quantum Algorithms and Machine Learning Reading Group
======
This is a reading material archive for a journal club held at Melbourne University. 

A paper will be sent out every fortnight. The goal is to openly discuss it at various depths as a group. So no matter what your background is, we will all hopefully learn something. The meetings are casual, so there's no pressure to present or anything like that.

In the past weeks we have covered the following papers:

### 1. Quantum Algorithm for Linear Systems of Equations - Seth Lloyd's group.
Sources: [PRL](https://doi.org/10.1103/PhysRevLett.103.150502) and [arXiv](https://arxiv.org/abs/0811.3171)

As the title suggests this paper introduces a new quantum algorithm for solving linear systems of equations. Classically, the best time scaling algorithm is roughly N sqrt(k) where k is something called the condition number (how sensitive the matrix is to changes). This new quantum algorithm scales roughly poly log(N) and poly (k), which is essentially exponentially faster. There is a catch though, the actual readout of the solution scales linearly (which is the same as classical). So the algorithm will be mainly beneficial for feeding the output into the input of some other quantum algorithm or approximating the expectation value.

The method uses a technique where a non-unitary transformation involving measurement is used within the quantum circuit. Computation is only continued if a particular qubit state is measured, otherwise it is repeated.

We later found that this idea could be extended to [linear differential equations](https://arxiv.org/abs/1010.2745)
and [non-linear differential equations](https://arxiv.org/abs/0812.4423)

### 2. Quantum Algorithms for Supervised and Unsupervised Machine Learning - Seth Lloyd's group.
Source: [arXiv](https://arxiv.org/abs/1307.0411)

This paper focuses on applying quantum algorithms on supervised and unsupervised clustering algorithms. 

The supervised algorithm (classification) is simply the process of assigning each new data point to a category with the closest mean. The closest means are found by applying a procedure called a swap test between a particular qubit register state representing the mean and an ancilla qudit (not qubit) representing the new datum. After that, the value of a normalisation constant, which can in principle be determined using Grover’s algorithm/quantum counting is used to .

The unsupervised algorithm is for k-means clustering which uses Lloyd’s algorithm (not related to the author of this paper). It’s a pretty cool algorithm, so you should check it out (there are a few visualisations of it around the net). The paper introduces the quantum Lloyd’s algorithm with speed up (although not exponential) over the classical version. Unfortunately, we didn’t get enough time to go through this section in detail.

I think the take home message of this paper was that quantum computers are powerful when it comes to manipulating large numbers of high-dimensional vector spaces, which is precisely what is required for vector-based machine learning. This paper runs through some examples of how existing quantum algorithms can be directly applied to supervised and unsupervised machine learning clustering problems.
