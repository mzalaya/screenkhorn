
# `SCREENKHORN`: Screening Sinkhorn Algorithm for Regularized optimal Transport

Implementation of SCREENKHORN algorithm from paper [Screening Sinkhorn Algorithm for Regularized Optimal Transport](https://arxiv.org/abs/1906.08540) in Python.

Package dependencies
====================
It requires the following Python packages:

- numpy
- scipy
- matplotlib
- autograd
- [POT](https://github.com/rflamary/POT)

Included modules
================
From a console or terminal clone the repository:
```
git clone https://github.com/mzalaya/screenkhorn
cd screenkhorn/
```
The folder contains the following files:
```
- screenkhorn.py:

# Toy example
- marge_expe.py
- marge_expe_v2.py

# Wasserstein discriminant analysis
- wda_screenkhorn.py:
- wda_expe:py

# Domain adaptation
- da_screenkhorn.py
- da_exp.py:

```

Small Demo
================
Given a ground metric `M`, the discrete measures `a` and `b`, and the entropy parameter `reg` that define the Sinkhorn divergence
distance. The parameters `n_budget` and `m_budget` correspond to the number of points to be considered. Then the Screenkhorn object can be created.

```python
>>> from screenkhorn import Screenkhorn 
>>> screenkhorn = Screenkhorn(a, b, M, reg, n_budget, m_budget, verbose=False)
>>> screen_lbfgsb = screenkhorn.lbfgsb()
>>> P_sc = screen_lbfgsb[2]
>>> # Screened marginals
>>> a_sc = P_sc @ np.ones(b.shape)
>>> b_sc = P_sc.T @ np.ones(a.shape)
```    

Citation
========
If you use this code, please cite:

```
@misc{alaya2019etal,
Author = {Alaya, Mokhtar Z. and  Bérar, Maxime and  Gasso, Gilles and  Rakotomamonjy, Alain},
Title = {Screening Sinkhorn Algorithm for Regularized Optimal Transport},
Year = {2019},
Eprint = {arXiv:1906.08540},
}
```