# 计算可靠度的程序_1
# import numpy as np
# import time
#
#
# # 改为矢量化计算能节省时间
#
# down_matrix = np.array([[0,0,0,0,2,2], [0,0,0,0,2,3], [1,0,0,0,2,3], [0,1,0,0,2,2], [0,1,0,0,3,2],
#                         [1,0,1,0,1,2], [1,0,1,0,1,3], [2,0,1,0,1,3], [2,0,2,0,0,2], [2,0,2,0,0,3],
#                         [3,0,2,0,0,3], [2,2,0,0,0,0], [2,2,0,0,0,1], [3,2,0,0,0,1], [2,3,0,0,0,0],
#                         [2,3,0,0,1,0], [1,1,0,0,1,1], [1,1,0,0,1,2], [2,1,0,0,1,2], [1,2,0,0,1,1],
#                         [1,2,0,0,2,1], [2,1,1,0,0,1], [3,1,1,0,0,1], [3,1,1,0,0,2], [0,1,0,1,2,1],
#                         [0,1,0,1,3,1], [0,2,0,1,3,1], [1,2,0,1,1,0], [1,3,0,1,1,0], [1,3,0,1,2,0],
#                         [0,2,0,2,2,0], [0,3,0,2,2,0], [0,3,0,2,3,0]])
# up_matrix = np.array([[3,0,3,3,3,2],[0,0,3,3,2,3],[3,0,0,3,2,3],[0,3,3,3,2,3],[0,3,3,0,3,2],
#                       [3,0,3,3,1,2],[1,0,3,3,1,3],[3,0,1,3,1,3],[3,0,3,3,0,2],[2,0,3,3,0,3],
#                       [3,0,2,3,0,3],[3,2,3,3,3,0],[2,2,3,3,0,3],[3,2,0,3,0,3],[2,3,3,3,0,3],
#                       [2,3,3,0,3,0],[3,1,3,3,3,1],[1,1,3,3,1,3],[3,1,0,3,1,3],[1,3,3,3,1,3],
#                       [1,3,3,0,3,1],[2,1,3,3,0,3],[3,1,3,3,0,1],[3,1,1,3,0,3],[0,3,3,3,2,1],
#                       [0,1,3,3,3,1],[0,3,3,1,3,1],[1,2,3,3,3,0],[1,3,3,3,1,0],[1,3,3,1,3,0],
#                       [0,2,3,3,3,0],[0,3,3,3,2,0],[0,3,3,2,3,0]])
# # down_matrix = np.array([[1,2,1,2,1], [1,2,1,1,2], [2,3,2,1,1]])
# # up_matrix = np.array([[3,3,3,3,3],[3,3,3,1,3],[3,3,3,1,1]])
# num = 6
# R = 0
# prob = np.ones((num, 4), dtype=int) * np.array([0.25,0.3,0.25,0.2])
# start_time = time.time()
# for i in range(down_matrix.shape[0]):
#     temp_p = np.zeros(num)
#     down = down_matrix[i, :]
#     up = up_matrix[i, :]
#     for j in range(temp_p.shape[0]):
#         temp_p[j] = np.sum(prob[j, (down[j]):(up[j] + 1)])
#     R += np.prod(temp_p)
# cal_time = time.time() - start_time
#
# print("d=2时的可靠度：", R)
# print("计算时间：", cal_time)


# 计算可靠度的程序_2
# import numpy as np
# import time
#
# # 输入矩阵
# # down_matrix = np.array([[0,0,0,0,2,2], [0,0,0,0,2,3], [1,0,0,0,2,3], [0,1,0,0,2,2], [0,1,0,0,3,2],
# #                         [1,0,1,0,1,2], [1,0,1,0,1,3], [2,0,1,0,1,3], [2,0,2,0,0,2], [2,0,2,0,0,3],
# #                         [3,0,2,0,0,3], [2,2,0,0,0,0], [2,2,0,0,0,1], [3,2,0,0,0,1], [2,3,0,0,0,0],
# #                         [2,3,0,0,1,0], [1,1,0,0,1,1], [1,1,0,0,1,2], [2,1,0,0,1,2], [1,2,0,0,1,1],
# #                         [1,2,0,0,2,1], [2,1,1,0,0,1], [3,1,1,0,0,1], [3,1,1,0,0,2], [0,1,0,1,2,1],
# #                         [0,1,0,1,3,1], [0,2,0,1,3,1], [1,2,0,1,1,0], [1,3,0,1,1,0], [1,3,0,1,2,0],
# #                         [0,2,0,2,2,0], [0,3,0,2,2,0], [0,3,0,2,3,0]])
# down_matrix = np.array([[2,1,1,2,1],[2,1,3,2,1],[2,1,1,1,2],[2,1,3,1,2]])
# # up_matrix = np.array([[3,0,3,3,3,2],[0,0,3,3,2,3],[3,0,0,3,2,3],[0,3,3,3,2,3],[0,3,3,0,3,2],
# #                       [3,0,3,3,1,2],[1,0,3,3,1,3],[3,0,1,3,1,3],[3,0,3,3,0,2],[2,0,3,3,0,3],
# #                       [3,0,2,3,0,3],[3,2,3,3,3,0],[2,2,3,3,0,3],[3,2,0,3,0,3],[2,3,3,3,0,3],
# #                       [2,3,3,0,3,0],[3,1,3,3,3,1],[1,1,3,3,1,3],[3,1,0,3,1,3],[1,3,3,3,1,3],
# #                       [1,3,3,0,3,1],[2,1,3,3,0,3],[3,1,3,3,0,1],[3,1,1,3,0,3],[0,3,3,3,2,1],
# #                       [0,1,3,3,3,1],[0,3,3,1,3,1],[1,2,3,3,3,0],[1,3,3,3,1,0],[1,3,3,1,3,0],
# #                       [0,2,3,3,3,0],[0,3,3,3,2,0],[0,3,3,2,3,0]])
# up_matrix = np.array([[3,3,2,3,3],[3,3,3,2,2],[3,3,2,1,3],[3,3,3,1,2]])
# # 参数设置
# num = 5
# R = 0
# # prob = np.ones((num, 4), dtype=int) * np.array([0.25, 0.3, 0.25, 0.2])
# prob = np.array([[0, 0.1, 0.2, 0.7], [0, 0.05, 0.75, 0.2], [0, 0.05, 0.65, 0.3],
#                  [0, 0.3, 0.5, 0.2], [0, 0.35, 0.4, 0.25]])
# start_time = time.time()
# for i in range(down_matrix.shape[0]):
#     down = down_matrix[i, :]
#     up = up_matrix[i, :]
#     # 计算 temp_p[j] = np.sum(prob[j, (down[j]):(up[j] + 1)]) 的向量化版本
#     temp_p = np.array([np.sum(prob[j, down[j]:(up[j] + 1)]) for j in range(num)])
#     R += np.prod(temp_p)
# end_time = time.time()
#
# print("可靠度：", R)
# print("耗时：", end_time - start_time)


# # test
# import networkx as nx
#
# G = nx.Graph()
# with open('bio-CE-PG.edges', 'r') as f:
#     for line in f:
#         v0, v1 = line.strip().split()
#         G.add_edge(v0, v1)

# import pandas as pd
#
# # 文件路径
# edges_file = 'bio-CE-PG.edges'
# # 读取 edges 文件
# edges = pd.read_csv(edges_file, sep=' ', header=None, names=['Source', 'Target', 'Weight'])
# # 打印读取的数据
# print(edges)


# # 检查空间是否被支配_程序1
# import numpy as np
#
#
# def mark_spaces(upper_bounds, lower_bounds):
#     marked_spaces = []
#     # 获取矩阵的行数
#     num_rows = upper_bounds.shape[0]
#     # 遍历每一对状态空间
#     for i in range(num_rows):
#         for j in range(num_rows):
#             if i != j:
#                 # 检查第i行是否完全小于等于或大于等于第j行
#                 if (np.all(upper_bounds[i] <= upper_bounds[j]) and np.all(lower_bounds[i] <= lower_bounds[j])) or \
#                    (np.all(upper_bounds[i] >= upper_bounds[j]) and np.all(lower_bounds[i] >= lower_bounds[j])):
#                     marked_spaces.append((lower_bounds[i], upper_bounds[i]))
#                     break
#     return marked_spaces
#
# # 示例数据
# # upper_bounds = np.array([
# #     [3, 2, 2, 3, 3],
# #     [3, 2, 3, 2, 2],
# #     [3, 3, 3, 2, 2],
# #     [3, 3, 2, 3, 1],
# #     [3, 3, 2, 3, 3],
# #     [3, 3, 3, 1, 2]
# # ])
# upper_bounds = np.array([
#     [3, 2, 3, 1, 3],
#     [3, 3, 3, 1, 2]
# ])
# # lower_bounds = np.array([
# #     [2, 1, 1, 2, 1],
# #     [2, 1, 3, 2, 1],
# #     [2, 3, 1, 2, 1],
# #     [2, 3, 1, 3, 1],
# #     [2, 1, 1, 1, 2],
# #     [2, 1, 3, 1, 2]
# # ])
# lower_bounds = np.array([
#     [1, 2, 1, 1, 2],
#     [1, 2, 1, 1, 2]
# ])
# marked_spaces = mark_spaces(upper_bounds, lower_bounds)
# print(marked_spaces)
# # 输出被标记的空间
# for lower, upper in marked_spaces:
#     print(f"下界: {lower}, 上界: {upper}")


# # 测试函数
# lower_bounds = np.array([
#     [2,1,1,2,1],
#     [2,1,3,2,1],
#     [2,3,1,2,1],
#     [2,3,1,3,1],
#     [2,1,1,1,2],
#     [2,1,3,1,2]
# ])
# # lower_bounds = np.load('Down_matrix.npy')
#
# upper_bounds = np.array([
#         [3,2,2,3,3],
#         [3,2,3,2,2],
#         [3,3,3,2,2],
#         [3,3,2,3,1],
#         [3,3,2,3,3],
#         [3,3,3,1,2],
# ])
# # upper_bounds = np.load('Up_matrix.npy')
#
# marked_spaces = mark_spaces(lower_bounds, upper_bounds)
# print("被标记的状态空间:")
# for space in marked_spaces:
#     print(f"下界: {space[0]}, 上界: {space[1]}")


# # 检查空间是否被支配_程序2
# import numpy as np
#
# # 定义上界和下界矩阵
# lower_bounds = np.array([[2,1,1,2,1],
#                          [2,1,3,2,1],
#                          [2,3,1,2,1],
#                          [2,3,1,3,1],
#                          [2,1,1,1,2],
#                          [2,1,3,1,2]])
# upper_bounds = np.array([[3,2,2,3,3],
#                          [3,2,3,2,2],
#                          [3,3,3,2,2],
#                          [3,3,2,3,1],
#                          [3,3,2,3,3],
#                          [3,3,3,1,2]])
#
# # 初始化标记矩阵
# marked_spaces = np.zeros(upper_bounds.shape[0], dtype=bool)
#
# # 获取矩阵维度
# num_spaces = upper_bounds.shape[0]
#
# # 遍历每个空间并进行比较
# for i in range(num_spaces):
#     for j in range(num_spaces):
#         if i != j and np.all(upper_bounds[i] <= upper_bounds[j]) and np.all(lower_bounds[i] >= lower_bounds[j]):
#             marked_spaces[i] = True
#             break  # 一旦找到一个匹配的j，i即被标记，不需要继续比较
#
# # 输出标记矩阵
# print("Marked Spaces Array:")
# print(marked_spaces)
#
# # 输出被标记的空间
# marked_indices = np.where(marked_spaces)[0]
# for idx in marked_indices:
#     print(f"Space {idx}:")
#     print(f"Upper Bounds: {upper_bounds[idx]}")
#     print(f"Lower Bounds: {lower_bounds[idx]}")

# import numpy as np
#
# # 创建几个NumPy数组
# array1 = np.array([1, 2, 3])
# array2 = np.array([4, 5, 6])
# array3 = np.array([7, 8, 9])
#
# # 将NumPy数组存储在列表中
# array_list = [array1, array2, array3]
#
# # 访问列表中的NumPy数组
# for array in array_list:
#     print(array)
# a = [array1, np.array([1, 2])]
# a[1][1]

# 已知的第一列值
first_column = [1, 2, 3, 4, 5]
# 使用None作为占位符表示第二列待定
pending_value = None
# 创建嵌套列表，第二列值待定
nested_list = [[value, pending_value] for value in first_column]
# 输出嵌套列表
nested_list[0][1] = 1
print(nested_list)


