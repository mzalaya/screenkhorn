
# `screenkhorn`: Screening Sinkhorn Algorithm
Implementation of SCREENKHORN algorithm from paper [Screening Sinkhorn Algorithm for Regularized Optimal Transport] (https://arxiv.org/abs/1906.08540) in Python.

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


Small Demo
================
Given a ground metric `M`, the discrete measures `a` and `b` and the entropy parameter `reg` that define the Wasserstein
metric. The parameters `n_budget` and `m_budget` correspond to the number of points to be considered. Then the Screenkhorn object can be created.

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