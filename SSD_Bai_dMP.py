# -*- coding: utf-8 -*-

import numpy as np
import time
# from loguru import logger
import logging
# logging.basicConfig(level=logging.INFO,filename='test_log.txt',filemode='w',format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logging.basicConfig(level=logging.DEBUG,filename='test_log.txt',filemode='w',format='%(asctime)s [line:%(lineno)d] %(levelname)s: %(message)s')
# logger.add("test_log.log")


def SSD(d_MP, Pr, m, mm):  # d_MP = np.array([[1,2,3],[1,2,3]])
    # 假设 d_MP 已经是一个 NumPy 数组
    NP = d_MP.shape[0]
    R = 0
    k = 1
    b0 = mm * np.ones(m, dtype=int)
    b = 0 * np.ones(m).astype(np.int32)
    b0_mat = np.zeros((100, m)).astype(np.int32)
    b_mat = np.zeros((100, m)).astype(np.int32)
    ind_mat = NP * np.ones((100, 1), dtype=int)
    T = np.zeros((10, 3))
    x = 0
    start_time = time.time()

    while k:
        ind = ind_mat[k - 1]
        w = 0
        for i in range(int(ind)):
            if sum(d_MP[i, :] <= b0) == m:
                temp = d_MP[i, :].copy()
                d_MP[i, :] = d_MP[w, :].copy()
                d_MP[w, :] = temp
                w += 1
        yy = d_MP[:w, :]
        nnp = yy.shape[0]
        y_pri = np.zeros(m)
        for j in range(m):
            y_pri[j] = np.min(yy[:, j])  # 求z*

        # 根据z*求关键下界向量v（一般两者相同）
        v = np.max(np.vstack((y_pri, b)), axis=0).astype(np.int32)

        # 根据Bai2018论文中的启发式规则计算所有d-MP的H，生成Hy数组，选择一个d-MP并求v0
        Hy = np.zeros(nnp).astype(np.int32)
        ww = np.zeros((nnp, m)).astype(np.int32)
        for l in range(nnp):
            for j in range(m):
                ww[l, j] = np.max([yy[l, j], b[j]]) - v[j]  # (b0-vl+1)*Pr
            Hy[l] = np.sum(ww[l, :])

        # 返回值H值最小的d-MP的索引
        Hy1 = np.where(Hy == np.min(Hy))[0]

        # 返回H值最小的d-MP向量
        yy1 = np.zeros((Hy1.size, m)).astype(np.int32)
        for z in range(Hy1.size):
            yy1[z, :m] = ww[Hy1[z], :]

        # 在合格d-MP zl中选择一个生成v0
        max_val, max_posi = np.min(Hy), np.argmin(Hy)
        v0 = np.max(np.vstack((yy[max_posi, :m], b)), axis=0)

        temp_p = np.zeros(m)
        for j in range(m):
            temp_p[j] = np.sum(Pr[j, (v0[j]):(b0[j] + 1)])
        R += np.prod(temp_p)
        # logger.debug("R = {}", R)
        # logging.debug("R is :{}".format(R))

        # 更新状态空间的上下边界矩阵b0, b
        s = 0
        a = np.zeros(m)
        for j in range(m):
            if v[j] < v0[j]:
                a[s] = j
                s += 1
        if s > 0:
            for d in range(s):
                for j in range(m):
                    if j == a[d]:
                        b0_mat[d + k - 1, j] = v0[j] - 1
                    else:
                        b0_mat[d + k - 1, j] = b0[j]
                    if j < a[d]:
                        b_mat[d + k - 1, j] = v0[j]
                    else:
                        b_mat[d + k - 1, j] = v[j]
                ind_mat[d + k - 1] = w
        k = k - 1 + s

        if k == 0:
            break
        else:
            # debug过程中发现:若不进行拷贝(.copy),则矩阵b0_mat变动,b0随之变动
            b0 = b0_mat[k - 1, :].copy()
            b = b_mat[k - 1, :].copy()
            x += 1
        logging.debug("b0 = {}".format(b0))
        logging.debug("x = {}".format(x))
    caltime = time.time() - start_time
    return R, caltime


if __name__ == '__main__':
    # ########### Example 1 ###########
    # dmp = np.array([[1,0,0,0,0,0,1,0,0,1,0],[1,0,0,1,0,0,0,1,0,1,0],[0,1,0,0,1,0,0,1,0,1,0],
    #                [0,1,1,0,0,1,0,1,0,1,0] ,[0,1,1,0,0,0,0,0,1,1,0] ,[0,1,1,0,0,0,0,0,0,0,1]])
    # prob = np.ones((11, 3)) * np.array([0.4, 0.3, 0.3])
    # m = 11
    # mm = 2
    # R, caltime = SSD(dmp, prob, m, mm)
    # print("可靠度:", R)
    ########### Example 2 ###########
    dmp = np.array([[3,2,1,0,0,1],[2,1,1,0,1,2],[2,2,0,0,1,1]])
    prob = np.array([[0.05,0.1,0.25,0.6],[0.1,0.3,0.6,0],[0.1,0.9,0,0],
                     [0.1,0.9,0,0],[0.1,0.9,0,0],[0.05,0.25,0.7,0]])
    m = 6
    mm = 3
    R, caltime = SSD(dmp, prob, m, mm)  # R = 0.6114149999999999
    print("可靠度:", R)
    print("计算时间:", caltime)

