bell_purification
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/bell_purification/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/bell_purification/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/bell_purification/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/bell_purification/branch/master)


Estimate the Bell state purification scheme performance and resource cost.

In this repo, we consider using the Deutsch protocol [[1]](#"Deutsch") for Bell state purification. We consider estimating the purification fidelity after $n$ nested successful purification rounds. The Bennett protocols [[2]](#"Bennett") is similar to Deutsch protocol, which is simulated using enforcing the middle step purification states to be Warner states.  More detailed information about nested purification can be found in Ref. [[3]](#"Dur").

## References
<a id="Deutsch">[1]</a> 
D. Deutsch, A. Ekert, R. Jozsa, C. Macchiavello, S. Popescu, and A. Sanpera, \"Quantum Privacy Amplification and the Security of Quantum Cryptography over Noisy Channels\", Phys. Rev. Lett. 77, 2818 (1996)

<a id="Bennett">[2]</a> 
C. H. Bennett, G. Brassard, S. Popescu, B. Schumacher,J. A. Smolin, and W. K. Wootters, \"Purification of Noisy Entanglement and Faithful Teleportation via Noisy Channels\", Phys. Rev. Lett. 76, 722 (1996)

<a id="Dur">[3]</a> 
W. Dur and H. J. Briegel, \"Entanglement purification and quantum error correction\", Reports on Progress in Physics 70, 1381 (2007).

### Copyright

Copyright (c) 2022, Chenxu Liu


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
