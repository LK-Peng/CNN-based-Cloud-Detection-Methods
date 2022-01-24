"""
    Date: 2020-12
    Function: 计算有效感受野，包括使用论文中原始方法及区分方向的半变异函数计算
    Ref:Luo W, Li Y, Urtasun R, et al. Understanding the effective receptive field in
        deep convolutional neural networks[J]. arXiv preprint arXiv:1701.04128, 2017.
"""

import numpy as np


def calculate_erf(img, h_ct, w_ct):
    img = np.sum(abs(img), axis=0)
    # square of pixel number (> (1-95.45) * center)
    erf = img > ((1 - 0.9545) * img[h_ct, w_ct])
    return erf.sum() ** 0.5

