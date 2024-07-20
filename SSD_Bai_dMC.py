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
    # b0 = 0 * np.ones(m, dtype=int)
    b0 = np.array([2, 1, 1, 1, 2])

    b0_matrix = np.empty((0, m), dtype=int)  # 存储下界
    v0_matrix = np.empty((0, m), dtype=int)  # 存储上界

    b_mat = np.zeros((100, m), dtype=int)
    b0_mat = np.empty((100, m), dtype=int)
    ind_mat = d_MC_num * np.ones((100, 1), dtype=int)
    x = 0
    start_time = time.time()

    while k:
        ind = ind_mat[k - 1]
        w = 0
        for i in range(int(ind)):
            # 将大于b0的d-MC放置在d-MC数组顶部
            if sum(d_MC[i, :] >= b0) == m:
                if i == w:
                    w += 1
                    continue
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
            for j in range(m):
                ww[l, j] = v[j] - np.min([yy[l, j], b[j]])  # 计算H(cl)
            Hy[l] = np.sum(ww[l, :])

        # 返回值H值最大的且相同的d-MC中的第一个索引
        Hy1 = np.where(Hy == np.max(Hy))[0]

        # 在合格d-MC cl中选择一个生成下边界向量v0
        # max_val, max_posi = np.max(Hy), np.argmax(Hy)
        # 基于d-MC分解也应该选择H(zl)最小的d-MC
        ax_val, max_posi = np.min(Hy), np.argmin(Hy)
        v0 = np.min(np.vstack((yy[max_posi, :m], b)), axis=0)

        # 记录由所有d-MC分解得到的所有空间的上下界向量矩阵
        v0_matrix = np.vstack([v0_matrix, v0])
        b0_matrix = np.vstack([b0_matrix, b0])

        # 计算合格空间的不可靠度
        temp_p = np.zeros(m)
        for j in range(m):
            temp_p[j] = np.sum(Pr[j, (b0[j]):(v0[j] + 1)])
        U += np.prod(temp_p)

        # 更新状态空间的上下边界向量矩阵b, b0
        s = 0
        a = np.zeros(m)
        for j in range(m):
            if v[j] > v0[j]:
                a[s] = j
                s += 1
        if s > 0:
            for d in range(s):
                for j in range(m):
                    if j == a[d]:
                        b0_mat[d + k - 1, j] = v0[j] + 1
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
            b = b_mat[k - 1]
            x += 1
    caltime = time.time() - start_time
    return 1-U, caltime, b0_matrix, v0_matrix

if __name__ == '__main__':
    ########### Example 2 ###########
    # # 2-MC
    # dmc = np.array([[1, 2, 1, 1, 1, 2],[3, 0, 1, 1, 1, 2],[3, 1, 1, 1, 1, 1],[3, 2, 1, 1, 1, 0],
    #                 [2, 2, 1, 1, 0, 2],[3, 2, 0, 1, 0, 2],[3, 1, 1, 1, 0, 2],[3, 1, 0, 1, 1, 2]])
    # # # 3-MC
    # # dmc = np.array([[2, 1, 1, 1, 2, 2],[3, 1, 1, 1, 1, 2],[3, 1, 1, 1, 2, 1],
    # #                 [3, 1, 0, 1, 2, 2],[3, 1, 1, 0, 2, 2]])
    # # prob = np.array([[0.05,0.1,0.25,0.6],[0.1,0.3,0.6,0],[0.1,0.9,0,0],
    # #                  [0.1,0.9,0,0],[0.1,0.9,0,0],[0.05,0.25,0.7,0]])
    # m = 6
    # mm = 3
    # R, caltime = SSD(dmc, prob, m, mm)  # R3 = 0.611415 R4 = 0.6
    # print("可靠度:", R)
    # print("计算时间:", caltime)
    ################################# My_Idea #################################
    m = 5
    mm = 3
    # # 1-MC 10个
    # dmc_1 = np.array([[3, 3, 3, 3, 0, 1],[3, 3, 3, 3, 1, 0],[0, 3, 1, 3, 3, 3],[1, 3, 0, 3, 3, 3],
    #                 [0, 0, 3, 3, 3, 1],[0, 1, 3, 3, 3, 0],[1, 0, 3, 3, 3, 0],[3, 3, 0, 0, 1, 3],
    #                 [3, 3, 0, 1, 0, 3],[3, 3, 1, 0, 0, 3]])
    # # 2-MC 18个
    # dmc_2 = np.array([[3, 0, 3, 3, 3, 2], [3, 1, 3, 3, 3, 1], [3, 2, 3, 3, 3, 0], [0, 3, 3, 3, 2, 3],
    #                 [1, 3, 3, 3, 1, 3], [2, 3, 3, 3, 0, 3], [0, 3, 3, 0, 3, 2], [0, 3, 3, 1, 3, 1],
    #                 [0, 3, 3, 2, 3, 0], [1, 3, 3, 0, 3, 1], [1, 3, 3, 1, 3, 0], [2, 3, 3, 0, 3, 0],
    #                 [3, 2, 0, 3, 0, 3], [3, 1, 1, 3, 0, 3], [3, 0, 2, 3, 0, 3], [3, 1, 0, 3, 1, 3],
    #                 [3, 0, 1, 3, 1, 3], [3, 0, 0, 3, 2, 3]])

    dmc = np.array([[3,3,2,1,3],[3,3,3,1,2]])
    prob = np.ones((m, 4), dtype=int) * np.array([0.275,0.275,0.25,0.2])
    R, cal_time, b0_mat, v0_mat = SSD(dmc, prob, m, mm)
    # R_1, caltime_1, _, _ = SSD(dmc_1, prob, m, mm)
    # R_2, caltime_2, _, _ = SSD(dmc_2, prob, m, mm)
    # R = R_1 - R_2
    # cal_time = caltime_1 + caltime_2
    # print("\n网络恰好满足需求水平d时的可靠度:", R)
    # print("计算时间:", cal_time)
    print("由d-MP分解得到的上界向量矩阵:\n", v0_mat)
    print("由d-MP分解得到的下界向量矩阵:\n", b0_mat)
