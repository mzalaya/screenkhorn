
#<font style="font-variant: small-caps">Screenkhorn</font>: Screening Sinkhorn Algorithm for Regularized Optimal Transport

Python implementation of SCREENKHORN algorithm from paper [Screening Sinkhorn Algorithm for Regularized Optimal Transport](https://arxiv.org/abs/1906.08540) (to appear in NeurIPS 2019).

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
- screenkhorn.py: Screenkhorn class

- marge_expe.py: Time comparison of Sinkhorn and Screenkhorn algorithms on a toy example
- marge_expe_v2.py

- wda_screenkhorn.py: Dimension reduction with screened optimal transport
- wda_expe.py: experiments

- da_screenkhorn.py: Domain adaptation with screened optimal transport
- da_exp.py: experiments

```

Small Demo
==========
Given a ground metric `C`, the discrete measures `a` and `b`, and the entropy parameter `reg` that define the Sinkhorn divergence
distance. The parameters `ns_budget` and `nt_budget` correspond to the number budget of points to be keeped in the source and the target domains, respectively. Then the Screenkhorn object is created.

```python
>>> from screenkhorn import Screenkhorn 
>>> screenkhorn = Screenkhorn(a, b, C, reg, ns_budget, nt_budget, verbose=False)

>>> # screened transportation plan 
>>> Psc = screenkhorn.lbfgsb()

>>> # screened marginals
>>> a_sc = Psc @ np.ones(b.shape)
>>> b_sc = Psc.T @ np.ones(a.shape)
```    

Citation
========
If you use `screenkhorn` in a scientific publication, we would appreciate citations. You can use the following bibtex entry:
```
@misc{alaya2019etal,
Author = {Alaya, Mokhtar Z. and  Bérar, Maxime and  Gasso, Gilles and  Rakotomamonjy, Alain},
Title = {Screening Sinkhorn Algorithm for Regularized Optimal Transport},
Year = {2019},
Eprint = {to appear in NeurIPS 2019},
}
```