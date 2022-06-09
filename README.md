# Nested H-score: Computing Maximal Correlation Functions with Deep Learning

This short script illustrates how to extract maximal correlation functions with deep learning. The key is to optimize a cost function, called **nested H-score**, a variant of the H-score introduced in [1][2][3].


## Maximal Correlation Functions ##
For given variables $X$, $Y$, the maximal correlation functions $(f_1^\ast(x), g_1^\ast(y)), \dots, (f_k^\ast(x), g_k^\ast(y)), \dots$ represent the maximal correlated aspects of $X$ and $Y$, which can be recursively defined as [4]

$$
f^\ast_i, g_i^\ast = \mathop{\arg \max}_{f_i, g_i} \ \mathbb{E}[f_i(X) g_i(Y)].
$$

where the maximization is over all $f_i, g_i$'s satisfying the orthongality constraints

$$
 \mathbb{E}[f_i(X)f_j^\ast(X)] = \mathbb{E}[g_i(Y)g_j^\ast(Y)] = \delta_{ij}, \quad\text{for all } 0 \leq j \leq i,
$$

and where $f_0^\ast = g_0^\ast \equiv 1$ are constant functions.

## H-score ##

Given *k*-dimensional features *f* of *X* and *g* of *Y*, the H-score <img src="https://render.githubusercontent.com/render/math?math=%5Cmathscr%7BH%7D(f%2C%20g)"> is defined as

<center>
<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle%5Cmathscr%7BH%7D(f%2C%20g)%20%3D%20%5Cmathbb%7BE%7D%5B%5Clangle%20f(X)%2C%20%20g(Y)%5Crangle%5D%20-%20%5Clangle%20%5Cmathbb%7BE%7D%5Bf(X)%5D%2C%20%20%5Cmathbb%7BE%7D%5Bg(Y)%5D%20%5Crangle%20-%20%5Cfrac%7B1%7D%7B2%7D%5Ccdot%20%5Cmathrm%7Btr%7D%5Cleft(%5Cmathbb%7BE%7D%5Bf(X)f%5E%7B%5Cmathrm%7BT%7D%7D(X)%5D%5Ccdot%20%5Cmathbb%7BE%7D%5Bg(Y)g%5E%7B%5Cmathrm%7BT%7D%7D(Y)%5D%5Cright)%0A">
</center>

It can be verified that [1] to maximize  <img src="https://render.githubusercontent.com/render/math?math=%5Cmathscr%7BH%7D(f%2C%20g)">, *f* and *g* should correspond to the *k*-dimensional subspace spanned by the top-*k* maximal correlation functions <img src="https://render.githubusercontent.com/render/math?math=(f_1%5E*%2C%20%5Cdots%2C%20f_k%5E*)"> and <img src="https://render.githubusercontent.com/render/math?math=(g_1%5E*%2C%20%5Cdots%2C%20g_k%5E*)">, respectively.

## Nested H-score ##

The nested H-score is the sum of H-scores associated with a series of nested features, defined as
<center>
<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle%5Cmathscr%7BH%7D%5E%7B%5Coplus%7D(f%2C%20g)%20%3D%20%5Csum_%7Bi%20%3D%201%7D%5Ek%20%5Cmathscr%7BH%7D(f%5E%7B%5Bi%5D%7D%2C%20g%5E%7B%5Bi%5D%7D)">
</center>

where <img src="https://render.githubusercontent.com/render/math?math=f%5E%7B%5Bi%5D%7D%20%5Ctriangleq%20%5Bf_1%2C%20%5Cdots%2C%20f_i%5D%5E%5Cmathrm%7BT%7D"> is the feature composed of the first *i*-dimensions of *f*.

Then, it can be verified that to maximize the nested H-score <img src="https://render.githubusercontent.com/render/math?math=%5Cmathscr%7BH%7D%5E%7B%5Coplus%7D(f%2C%20g)">, the one-dimension features <img src="https://render.githubusercontent.com/render/math?math=f_i"> and <img src="https://render.githubusercontent.com/render/math?math=g_i"> must be aligned to the *i*-th maximal correlation functions <img src="https://render.githubusercontent.com/render/math?math=f_i%5E*">, <img src="https://render.githubusercontent.com/render/math?math=g_i%5E*">, respectively.

More precisely, the functions <img src="https://render.githubusercontent.com/render/math?math=f%20%3D%20(f_1%2C%20%5Cdots%2C%20f_k)%2C%20g%20%3D%20(g_1%2C%20%5Cdots%2C%20g_k)"> that maximize the nested H-score <img src="https://render.githubusercontent.com/render/math?math=%5Cmathscr%7BH%7D%5E%7B%5Coplus%7D(f%2C%20g)"> would satisfy

<center>
<img src="https://render.githubusercontent.com/render/math?math=f_i%20%3D%20a_i%20%5Ccdot%20f_i%5E*%2C%20%5Cquad%20g_i%20%20%3D%20b_i%20%5Ccdot%20g_i%5E*">
</center>
where <img src="https://render.githubusercontent.com/render/math?math=a_i%2C%20b_i"> are scalars with <img src="https://render.githubusercontent.com/render/math?math=a_i%20%5Ccdot%20b_i%20%3D%20%5Cmathbb%7BE%7D%5Bf_i%5E*(X)%20g_i%5E*(Y)%5D">.


With "<img src="https://render.githubusercontent.com/render/math?math=%2B%5C!%5C!%5C!%5C!%2B">" denoting the feature concatenation operation, we can compute nested H-score <img src="https://render.githubusercontent.com/render/math?math=%5Cmathscr%7BH%7D%5E%7B%5Coplus%7D(f%2C%20g)"> with the following nested structure.

<center>
<img src="images/nested_H.png" width="768">
</center>


---

## References ##

[1] Wang, Lichen, Jiaxiang Wu, Shao-Lun Huang, Lizhong Zheng, Xiangxiang Xu, Lin Zhang, and Junzhou Huang. "[An efficient approach to informative feature extraction from multimodal data](https://ojs.aaai.org/index.php/AAAI/article/view/4464)." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, no. 01, pp. 5281-5288. 2019.

[2] Xu, Xiangxiang, and Shao-Lun Huang. "[Maximal correlation regression](https://ieeexplore.ieee.org/abstract/document/8979352)." IEEE Access 8 (2020): 26591-26601.

[3] Xu, Xiangxiang, Shao-Lun Huang, Lizhong Zheng, and Gregory W. Wornell. 2022. "[An Information Theoretic Interpretation to Deep Neural Networks](https://www.mdpi.com/1099-4300/24/1/135)" Entropy 24, no. 1: 135.

[4] Huang, Shao-Lun, Anuran Makur, Gregory W. Wornell, and Lizhong Zheng. "[On universal features for high-dimensional learning and inference](https://arxiv.org/pdf/1911.09105.pdf)." arXiv preprint arXiv:1911.09105 (2019).

