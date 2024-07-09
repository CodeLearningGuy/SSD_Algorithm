# -*- coding: utf-8 -*-
import numpy as np
import time
import logging
logging.basicConfig(level=logging.DEBUG,filename='test_log.txt',filemode='w',format='%(asctime)s [line:%(lineno)d] %(levelname)s: %(message)s')
from scipy.io import loadmat


def up_down_decom(d_MP, d_MC, Pr, num, cap):
    # 网络恰好满足需求水平d时的可靠度
    R = 0
    # num是组件（边）数, cap是所有组件一致的最大容量
    d_mp = d_MP.copy()
    # 最终分解得到的所有空间的下界和上界向量矩阵
    down_matrix = np.empty((0, num), dtype=int)
    up_matrix = np.empty((0, num), dtype=int)
    # d-MP和d-MC数量
    n_d_MP = d_MP.shape[0]
    n_d_MC = d_MC.shape[0]
    # 状态空间分解次数
    x = 0
    # 状态空间数
    num_space = 0

    # 首先,利用d-MP分解每个d-MC得到每个d-MP对应的上界向量

    # 给d-MP编号,每个d-MP都有一个属于它的编号,从0到n_d_MP
    d_MP_index = np.arange(n_d_MP, dtype=int)
    # 初始化字典存储每个d-MP分解得到的多个上界
    d_MP_up_matrix = {idx: np.empty((0, num), dtype=int) for idx in d_MP_index}

    for i in range(n_d_MC):
        b0 = d_MC[i, :]
        b = 0 * np.ones(num).astype(np.int32)
        b0_mat = np.zeros((100, num)).astype(np.int32)
        b_mat = np.zeros((100, num)).astype(np.int32)

        ind_mat = n_d_MP * np.ones(100, dtype=int)
        # k表示此次分解得到的空间数量
        k = 1
        while k:
            ind = ind_mat[k - 1]
            num_qlfd = 0  # 合格d_MP的数量(小于d_MC_i的d_MP的数量)
            for j in range(ind):
                if np.all(d_mp[j, :] <= b0):
                    # 目的是在d-MP都合格时省去交换位置的操作,但不一定提高效率
                    # if j == num_qlfd:
                    #     num_qlfd += 1
                    #     continue
                    temp_ind = d_MP_index[j].copy()
                    d_MP_index[j] = d_MP_index[num_qlfd].copy()
                    d_MP_index[num_qlfd] = temp_ind

                    temp = d_mp[j, :].copy()
                    d_mp[j, :] = d_mp[num_qlfd, :].copy()
                    d_mp[num_qlfd, :] = temp
                    num_qlfd += 1
            qlfd_d_MP_ind = d_MP_index[:num_qlfd]
            qlfd_d_MP = d_mp[:num_qlfd, :]

            # 求d-MP最小值组成的z*
            z_star = np.min(qlfd_d_MP, axis=0)
            # 关键下界向量v（一般和z*一致）, v<v0
            # v = z_star

            v = np.max(np.vstack((z_star, b)), axis=0)

            # 根据启发式规则计算所有d-MP的H值,然后选择一个d-MP求v0, v<v0
            difference_matrix = qlfd_d_MP - v
            H = np.sum(difference_matrix, axis=1)
            # 返回值H值最小的合格d-MP索引（可能多于一个）
            H_min_ind = np.where(H == np.min(H))[0]

            # # 返回H值最小的合格d-MP（可能多于一个）和v的差值矩阵,改用新的启发式规则时才使用
            # diff_matrix = np.zeros((H_min_ind.size, num), dtype=int)
            # for z in range(H_min_ind.size):
            #     diff_matrix[z, :] = difference_matrix[H_min_ind[z], :]

            # 任意选择（第一个）一个H值最小的合格d-MP作为v0, v<v0
            sel_H_min_ind = H_min_ind[0]
            # v0 = qlfd_d_MP[sel_H_min_ind, :]
            v0 = np.max(np.vstack((qlfd_d_MP[sel_H_min_ind, :], b)), axis=0)

            # print(v0)

            sel_d_MP_ind = qlfd_d_MP_ind[sel_H_min_ind]
            # 记录经过多次分解后每个d-MP对应的多个上界向量
            d_MP_up_matrix[sel_d_MP_ind] = np.vstack((d_MP_up_matrix[sel_d_MP_ind], b0))
            num_space += 1

            # print(b0)

            # 更新状态空间的上边界矩阵b0_mat
            a = np.where(v < v0)[0]
            s = a.shape[0]
            if s > 0:
                for d in range(s):
                    for y in range(num):
                        if y == a[d]:
                            b0_mat[d + k - 1, y] = v0[y] - 1
                        else:
                            b0_mat[d + k - 1, y] = b0[y]
                        if y < a[d]:
                            b_mat[d + k - 1, y] = v0[y]
                        else:
                            b_mat[d + k - 1, y] = v[y]
                    ind_mat[d + k - 1] = num_qlfd
            k = k - 1 + s
            if k == 0:
                break
            else:
                b0 = b0_mat[k - 1, :].copy()
                b = b_mat[k - 1, :].copy()
                x += 1
    # print("每个d-MP对应的多个上界向量矩阵\n", d_MP_up_matrix)
    # print("空间分解得到的空间数量", nums_pace)

    # 再利用得到的上界分解每个d-MP
    for i in range(n_d_MP):
        b = cap * np.ones(num, dtype=int)
        # b = np.array([3, 2, 1, 2, 2], dtype=int)

        b0 = d_MP[i, :]
        b_mat = np.zeros((100, num), dtype=int)
        b0_mat = np.zeros((100, num), dtype=int)
        up_vec = d_MP_up_matrix[i]
        if up_vec.size == 0:
            break
        ind_mat = up_vec.shape[0] * np.ones(100, dtype=int)
        # 循环次数
        num_while = 0
        # k表示此次分解得到的空间数量
        k = 1
        while k:
            if num_while != 0:
                ind = ind_mat[k - 1]
                num_qlfd = 0  # 合格up_vector数(大于b0的up_vector数)
                for j in range(ind):
                    if np.all(up_vec[j, :] >= b0):
                        temp = up_vec[j, :].copy()
                        up_vec[j, :] = up_vec[num_qlfd, :].copy()
                        up_vec[num_qlfd, :] = temp
                        num_qlfd += 1
                qlfd_up_vec = up_vec[:num_qlfd, :]
            else:
                num_qlfd = up_vec.shape[0]
                qlfd_up_vec = up_vec.copy()
            num_while += 1

            # 求上界最大值组成的c*
            c_star = np.max(qlfd_up_vec, axis=0)
            # 根据c*求关键上界向量v, v>v0
            # v = c_star  # 不考虑b会出现错误
            v = np.min(np.vstack((c_star, b)), axis=0)
            # 根据启发式规则计算所有up_vector的H值,然后选择一个求v0
            difference_matrix_2 = v - qlfd_up_vec
            H_2 = np.sum(difference_matrix_2, axis=1)
            # 返回值H值最小的up_vector索引（可能多于一个）
            H_2_min_ind = np.where(H_2 == np.min(H_2))[0]
            # 任意选择（第一个）一个H值最小的up_vector作为v0, v>v0
            sel_H_min_ind = H_2_min_ind[0]
            # v0 = qlfd_up_vec[sel_H_min_ind, :]  # 不考虑b会出现错误
            v0 = np.min(np.vstack((qlfd_up_vec[sel_H_min_ind, :], b)), axis=0)
            print(b0)
            print(v0)

            # 记录由所有d-MC分解得到的所有空间的上下界向量矩阵
            up_matrix = np.vstack([up_matrix, v0])
            down_matrix = np.vstack([down_matrix, b0])
            num_space += 1

            # 更新状态空间的下边界矩阵b0_mat
            a = np.where(v > v0)[0]
            s = a.shape[0]
            if s > 0:
                for d in range(s):
                    for j in range(num):
                        if j == a[d]:
                            b0_mat[d + k - 1, j] = v0[j] + 1
                        else:
                            b0_mat[d + k - 1, j] = b0[j]
                        if j < a[d]:
                            b_mat[d + k - 1, j] = v0[j]
                        else:
                            b_mat[d + k - 1, j] = v[j]
                    ind_mat[d + k - 1] = num_qlfd
            k = k - 1 + s

            if k == 0:
                break
            else:
                b0 = b0_mat[k - 1, :].copy()
                b = b_mat[k - 1].copy()
                x += 1

    for i in range(down_matrix.shape[0]):
        down = down_matrix[i, :]
        up = up_matrix[i, :]
        # 计算所有空间的概率的向量化版本
        temp_p = np.array([np.sum(Pr[j, down[j]:(up[j] + 1)]) for j in range(num)])
        # print("\n", np.prod(temp_p))
        R += np.prod(temp_p)

    # print("空间下界向量矩阵\n", down_matrix)
    # print("空间上界向量矩阵\n", up_matrix)
    # print("空间分解得到的空间总数量", num_space)
    return R, num_space, down_matrix, up_matrix


if __name__ == '__main__':
    m = 5  # 组件数
    c = 3  # 组件最大容量
    ################# 白光晗_2020_TR_基于状态空间分解的两端多状态网络可靠性评估改进方法 m=6 c=3 #################
    # # 2-MP 9个
    # d_mp = np.array([[2, 2, 0, 0, 0, 0], [1, 2, 0, 1, 1, 0], [0, 2, 0, 2, 2, 0], [1, 1, 0, 0, 1, 1],
    #                 [0, 1, 0, 1, 2, 1], [2, 1, 1, 0, 0, 1], [0, 0, 0, 0, 2, 2], [1, 0, 1, 0, 1, 2],
    #                 [2, 0, 2, 0, 0, 2]])
    # # 2-MC 18个
    # d_mc = np.array([[3, 0, 3, 3, 3, 2], [3, 1, 3, 3, 3, 1], [3, 2, 3, 3, 3, 0], [0, 3, 3, 3, 2, 3],
    #                 [1, 3, 3, 3, 1, 3], [2, 3, 3, 3, 0, 3], [0, 3, 3, 0, 3, 2], [0, 3, 3, 1, 3, 1],
    #                 [0, 3, 3, 2, 3, 0], [1, 3, 3, 0, 3, 1], [1, 3, 3, 1, 3, 0], [2, 3, 3, 0, 3, 0],
    #                 [3, 2, 0, 3, 0, 3], [3, 1, 1, 3, 0, 3], [3, 0, 2, 3, 0, 3], [3, 1, 0, 3, 1, 3],
    #                 [3, 0, 1, 3, 1, 3], [3, 0, 0, 3, 2, 3]])
    # 组件状态分布矩阵
    # prob = np.ones((m, 4), dtype=int) * np.array([0.275, 0.275, 0.25, 0.2])
    ################################# Liu-Tao-Fig-2.1(与上例一致) m=6 c=3 #################################
    # # 1-MP 4个
    # d_mp = np.array([[0, 0, 0, 0, 1, 1], [0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1], [1, 1, 0, 0, 0, 0]])
    # # 1-MC 8个
    # d_mc = np.array([[3, 0, 1, 1, 1, 1], [3, 1, 1, 1, 1, 0], [0, 2, 1, 1, 1, 2], [1, 2, 1, 1, 0, 2],
    #                 [3, 1, 0, 1, 0, 2], [3, 0, 1, 1, 0, 2], [3, 0, 0, 1, 1, 2], [1, 2, 1, 0, 1, 0]])
    # prob = np.array([[0.05, 0.1, 0.25, 0.6], [0.1, 0.3, 0.6, 0], [0.1, 0.9, 0, 0],
    #                  [0.1, 0.9, 0, 0], [0.1, 0.9, 0, 0], [0.05, 0.25, 0.7, 0]])
    ################################# Liu-Tao-Fig-2.1(与上例一致) m=6 c=3 #################################
    # # 3-MP 5个
    # d_mp = np.load('Bridge_3_MP_Mat.npy')
    # # 3-MC 6个
    # d_mc = np.array([[1, 2, 1, 2, 2], [3, 2, 1, 0, 2], [3, 1, 1, 2, 2],
    #                  [3, 2, 1, 2, 1], [2, 2, 1, 1, 2], [3, 2, 0, 1, 2]])
    # prob = np.array([[0.002, 0.013, 0.125, 0.86], [0.005, 0.01, 0.985, 0], [0.11, 0.89, 0, 0],
    #                  [0.003, 0.012, 0.985, 0], [0.006, 0.015, 0.979, 0]])
    ################################# 许贝师姐大论文-ARPA网络（无向）-图5.2-6节点9边 m=9 c=10 #################################
    # # 1-MP 13个
    # d_mp = np.load('ARPA_1_MP_Mat.npy')
    # # 1-MC 31个
    # mat = loadmat('ARPA_1_MC_Mat.mat')
    # # # 2-MP 59个
    # # d_mp = np.load('ARPA_2_MP_Mat.npy')
    # # # 2-MC 74个
    # # mat = loadmat('ARPA_2_MC_Mat.mat')
    #
    # Omega = mat['Omega']
    # d_mc = np.array(Omega)
    # # row_vector_3sf = np.array([0.015, 0.030, 0.045, 0.061, 0.076, 0.091, 0.106, 0.121, 0.136, 0.152, 0.167])
    # row_vector_3sf = np.array([0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # prob = np.tile(row_vector_3sf, (9, 1))
    ################################# LYK_2022_递归不相交和评估最大最小容量向量多状态网络可靠性 m=5 c=3 #################################
    d_mp = np.array([[2, 1, 1, 2, 1], [2, 1, 1, 1, 2], [2, 1, 1, 1, 3]])
    d_mc = np.array([[3, 2, 2, 3, 3], [3, 3, 2, 3, 3], [3, 3, 3, 2, 2]])
    prob = np.array([[0, 0.1, 0.2, 0.7], [0, 0.05, 0.75, 0.2], [0, 0.05, 0.65, 0.3],
                     [0, 0.3, 0.5, 0.2], [0, 0.35, 0.4, 0.25]])

    start_time = time.time()
    Reliability, Num_space, Down_matrix, Up_matrix = up_down_decom(d_mp, d_mc, prob, m, c)
    Calcu_time = time.time() - start_time
    print("空间下界向量矩阵 Down_matrix: \n", Down_matrix)
    print("空间上界向量矩阵 Up_matrix: \n", Up_matrix)
    print("空间分解过程中的空间总数:", Num_space)
    print("网络恰好满足需求水平d时的可靠度:", Reliability)
    print("计算时间:", Calcu_time)

