import numpy as np
import time


# 为无向边添加对称性
def make_undirected(edge_index):
    new_index = edge_index.copy()
    for (u, v), idx in edge_index.items():
        if (v, u) not in new_index:
            new_index[(v, u)] = idx
    return new_index


def distance(L):
    """
    递归算法计算每个节点到汇点的最小距离
    """
    n = len(L)
    Ld = [-1] * n
    queue = [i for i in range(n) if -1 in L[i]]
    for node in queue:
        Ld[node] = 0

    while queue:
        node = queue.pop(0)
        for neighbor in L[node]:
            if neighbor != -1 and (Ld[neighbor] == -1 or Ld[neighbor] > Ld[node] + 1):
                Ld[neighbor] = Ld[node] + 1
                queue.append(neighbor)
    return Ld


def find_paths(L, Ld, Q, u, P, S, all_paths):
    """
    递归函数寻找所有最小路径
    """
    if u == -1:
        all_paths.append(P.copy())
        return

    for v in L[u]:
        if v == -1:
            P.append(v)
            find_paths(L, Ld, Q, v, P, S, all_paths)
            P.pop()
        elif v not in P:
            k = Ld[v]
            if S[k] < Q[k]:
                P.append(v)
                S[k] += 1
                find_paths(L, Ld, Q, v, P, S, all_paths)
                S[k] -= 1
                P.pop()


def convert_to_edges(paths, edge_index):
    edge_paths = []
    for path in paths:
        edge_path = []
        for i in range(len(path) - 1):
            edge_path.append(edge_index[(path[i], path[i + 1])])
        edge_paths.append(edge_path)
    return edge_paths


def create_matrix(edge_paths, num_edges):
    num_paths = len(edge_paths)
    matrix = np.zeros((num_paths, num_edges), dtype=int)
    for i, path in enumerate(edge_paths):
        for edge in path:
            matrix[i, edge - 1] = 1  # 边编号从1开始，矩阵索引从0开始
    return matrix


if __name__ == "__main__":
    # # Example-1-白老师2016-Fig.2
    # # 节点索引的链接路径结构
    # L = {
    #     0: [1, 2], 1: [2, 3], 2: [1, 3, 4],
    #     3: [2, 4, -1], 4: [2, 3, -1]
    # }
    # # 无向边编号
    # edge_index = {
    #     (0, 1): 1, (0, 2): 2, (1, 2): 3, (1, 3): 4, (2, 3): 5,
    #     (2, 4): 6, (3, 4): 7, (3, -1): 8, (4, -1): 9
    # }

    # # Example-2-许贝师姐大论文-3*3网格网络（无向）-Fig.5.3
    # # 节点索引的链接路径结构
    # L = {
    #     0: [1, 3], 1: [2, 4], 2: [1, 5],
    #     3: [4, 6], 4: [1, 3, 5, 7], 5: [2, 4, -1],
    #     6: [3, 7], 7: [4, 6, -1]
    # }
    # # 无向边编号
    # edge_index = {
    #     (0, 1): 1, (1, 2): 2, (0, 3): 3, (1, 4): 4, (2, 5): 5,
    #     (3, 4): 6, (4, 5): 7, (3, 6): 8, (4, 7): 9, (5, -1): 10,
    #     (6, 7): 11, (7, -1): 12
    # }

    # # Example-3-桥网络（有向）
    # # 节点索引的链接路径结构
    # L = {
    #     0: [1, 2], 1: [2, -1], 2: [1, -1]
    # }
    # # 边编号
    # edge_index = {
    #     (0, 1): 1, (0, 2): 2, (2, 1): 3, (1, 2): 4, (1, -1): 5, (2, -1): 6
    # }

    # # Example-4-许贝师姐大论文-4*4网格网络（无向）-Fig.5.3
    # # 节点索引的链接路径结构
    # L = {
    #     0: [1, 3], 1: [2, 4], 2: [1, 5],
    #     3: [4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8],
    #     6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, -1],
    #     9: [6, 10], 10: [7, 9, -1]
    # }
    # # 无向边编号
    # edge_index = {
    #     (0, 1): 1, (1, 2): 2, (0, 3): 3, (1, 4): 4, (2, 5): 5,
    #     (3, 4): 6, (4, 5): 7, (3, 6): 8, (4, 7): 9, (5, 8): 10,
    #     (6, 7): 11, (7, 8): 12, (6, 9): 13, (7, 10): 14,
    #     (8, -1): 15, (9, 10): 16, (10, -1): 17
    # }

    # # Example-5-白老师2018-Fig.2-11节点21边
    # L = {
    #     0: [1, 2, 3, 4], 1: [5], 2: [4, 7, 9],
    #     3: [1, 5, 6], 4: [3, 6, 7], 5: [8],
    #     6: [10], 7: [6], 8: [10],
    #     9: [6, 10], 10: [-1]
    # }
    # edge_index = {
    #     (0, 1): 1, (3, 1): 2, (0, 2): 3, (0, 3): 4, (4, 3): 5,
    #     (0, 4): 6, (2, 4): 7, (1, 5): 8, (3, 5): 9, (3, 6): 10,
    #     (4, 6): 11, (7, 6): 12, (9, 6): 13, (2, 7): 14,
    #     (4, 7): 15, (5, 8): 16, (2, 9): 17, (6, 10): 18,
    #     (8, 10): 19, (9, 10): 20, (10, -1): 21
    # }

    # # Example-6-许贝师姐大论文-ARPA网络（无向）-Fig.5.2-6节点9边
    # # 节点索引的链接路径结构
    # L = {
    #     0: [1, 2], 1: [2, 3], 2: [1, 3, 4], 3: [1, 2, 4, -1], 4: [2, 3, -1]
    # }
    # # 边编号
    # edge_index = {
    #     (0, 1): 1, (0, 2): 2, (1, 2): 3, (1, 3): 4, (2, 3): 5, (2, 4): 6,
    #     (3, 4): 7, (3, -1): 8, (4, -1): 9
    # }

    # Example-3-桥网络（无向）
    # 节点索引的链接路径结构
    L = {
        0: [1, 2], 1: [2, -1], 2: [1, -1]
    }
    # 边编号
    edge_index = {
        (0, 1): 1, (0, 2): 4, (1, 2): 3, (1, -1): 2, (2, -1): 5
    }

    # 记录开始时间
    start_time = time.time()
    # 更新后的边编号，注意：有向网络可以注释掉这一步
    edge_index_aft = make_undirected(edge_index)
    # 标记每个节点到汇点的最小距离
    Ld = distance(L)
    # 获取每个距离对应的节点数量
    Q = [0] * (max(Ld) + 1)
    for d in Ld:
        Q[d] += 1
    # 初始化
    u = 0
    P = [u]  # 路径从节点0开始
    S = [0] * (max(Ld) + 1)
    all_paths = []
    # 执行算法
    find_paths(L, Ld, Q, u, P, S, all_paths)
    # 转换路集表示，注意：有向网络直接使用edge_index
    # edge_paths = convert_to_edges(all_paths, edge_index)
    edge_paths = convert_to_edges(all_paths, edge_index_aft)
    # 创建最小路集矩阵
    num_edges = len(edge_index)
    matrix = create_matrix(edge_paths, num_edges)
    # 记录结束时间
    caltime = time.time() - start_time

    print("所有最小路集的边表示:")
    for edge_path in edge_paths:
        print(edge_path)
    print("\n路集矩阵:")
    print(matrix)
    print("网络中的最小路集数量:", matrix.shape[0])
    print("计算时间:", caltime)

