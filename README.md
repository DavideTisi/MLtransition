# MLtransition
repo with the code for the phase transition drove by descriptor difference.

The code used for the study of the transition path from the gamma to beta phase of Li3PS4.
The main idea is to minimize a loss function which depends on the difference between the descriptor of beta and the descriptor of the enviroment, and a term to keep the energy in check.
The loss function is minimized with the [Nelder-Mead](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method).
Very early stage of development. 

The code relies on librascal, equistore and rascaline
