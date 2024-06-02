# -*- coding: utf-8 -*-

import numpy as np
import time
import logging
logging.basicConfig(level=logging.DEBUG,filename='test_log.txt',filemode='w',format='%(asctime)s [line:%(lineno)d] %(levelname)s: %(message)s')


def SSD(d_MC, Pr, m, mm):  # m是组件数，mm是组件容量
    d_MC_num = d_MC.shape[0]
    U = 0
    k = 1
    b = mm * np.ones(m, dtype=int)
    b0 = 0 * np.ones(m, dtype=int)
    b_mat = np.empty((0, m), dtype=int)
    b0_mat = np.empty((0, m), dtype=int)
    ind_mat = d_MC_num * np.ones((100, 1), dtype=int)
    x = 0
    start_time = time.time()

    while k:
        ind = ind_mat[k - 1]
        w = 0
        for i in range(int(ind)):
            # 将大于b0的d-MC放置在d-MC数组顶部
            if sum(d_MC[i, :] >= b0) == m:
                temp = d_MC[i, :].copy()
                d_MC[i, :] = d_MC[w, :].copy()
                d_MC[w, :] = temp
                w += 1
        yy = d_MC[:w, :]
        nnp = yy.shape[0]
        y_pri = np.zeros(m, dtype=int)
        for j in range(m):
            y_pri[j] = np.max(yy[:, j])  # 求c*

        # 根据c*求关键上界向量v（一般两者相同）
        v = np.min(np.vstack((y_pri, b)), axis=0)

        # 根据Bai2018论文中的启发式规则计算所有d-MC的H，生成Hy数组，选择一个d-MC并求下界向量v0
        Hy = np.zeros(nnp, dtype=int)
        ww = np.zeros((nnp, m), dtype=int)
        for l in range(nnp):
            v0_pri = np.max(np.vstack((yy[l, :], b)), axis=0)
            for j in range(m):
                ww[l, j] = np.max([yy[l, j], b[j]]) - v[j]  # (b0-vl+1)*Pr
            Hy[l] = np.sum(ww[l, :])

        # 返回值H值最小的d-MP的索引
        Hy1 = np.where(Hy == np.min(Hy))[0]
    print(Hy1)
    print(Hy1)
    print(Hy1)
    print(Hy)
    print(Hy1)