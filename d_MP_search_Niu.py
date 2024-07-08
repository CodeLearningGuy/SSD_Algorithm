import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from networkx.algorithms.flow import preflow_push


# 读取网络数据
def read_network():
    G = nx.Graph()
    # Example-1-NYF-2020-Fig-2（无向桥网络）
    # G.add_edge('s', '1', capacity=3)
    # G.add_edge('1', 't', capacity=2)
    # G.add_edge('1', '2', capacity=1)
    # G.add_edge('s', '2', capacity=2)
    # G.add_edge('2', 't', capacity=2)
    # Example-2-许贝师姐大论文-ARPA网络（无向）-Fig.5.2-6节点9边
    # 1
    # G.add_edge('s', '1', capacity=7)
    # G.add_edge('s', '2', capacity=5)
    # G.add_edge('1', '2', capacity=4)
    # G.add_edge('1', '3', capacity=6)
    # G.add_edge('2', '3', capacity=4)
    # G.add_edge('2', '4', capacity=5)
    # G.add_edge('3', '4', capacity=4)
    # G.add_edge('3', 't', capacity=6)
    # G.add_edge('4', 't', capacity=7)
    # 2
    G.add_edge('s', '1', capacity=10)
    G.add_edge('s', '2', capacity=10)
    G.add_edge('1', '2', capacity=10)
    G.add_edge('1', '3', capacity=10)
    G.add_edge('2', '3', capacity=10)
    G.add_edge('2', '4', capacity=10)
    G.add_edge('3', '4', capacity=10)
    G.add_edge('3', 't', capacity=10)
    G.add_edge('4', 't', capacity=10)
    # Example-3-牛义锋2020-Fig.4（无向）-16节点20边
    # G.add_edge('s', '1', capacity=4)
    # G.add_edge('s', '4', capacity=4)
    # G.add_edge('1', '2', capacity=4)
    # G.add_edge('1', '5', capacity=4)
    # G.add_edge('2', '3', capacity=4)
    # G.add_edge('3', '7', capacity=4)
    # G.add_edge('4', '5', capacity=4)
    # G.add_edge('4', '8', capacity=4)
    # G.add_edge('5', '6', capacity=4)
    # G.add_edge('5', '9', capacity=4)
    # G.add_edge('6', '10', capacity=4)
    # G.add_edge('7', '11', capacity=4)
    # G.add_edge('8', '12', capacity=4)
    # G.add_edge('9', '10', capacity=4)
    # G.add_edge('10', '11', capacity=4)
    # G.add_edge('10', '14', capacity=4)
    # G.add_edge('11', 't', capacity=4)
    # G.add_edge('12', '13', capacity=4)
    # G.add_edge('13', '14', capacity=4)
    # G.add_edge('14', 't', capacity=4)
    return G


# 绘制网络图
def draw_network(G):
    pos = nx.spring_layout(G)  # 布局算法选择
    edge_labels = nx.get_edge_attributes(G, 'capacity')
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=14, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)
    plt.title("Network Graph")
    plt.show()


# 示例运行
G = read_network()
draw_network(G)


# 最大流算法
def max_flow(G, source, sink):
    # flow_value, _ = nx.maximum_flow(G, source, sink)
    flow_value = preflow_push(G, source, sink)
    # return flow_value
    return flow_value.graph["flow_value"]


# 计算Li值
def calculate_Li(G, d):
    Li = {}
    for edge in G.edges(data=True):
        u, v, data = edge
        # capacity = data['capacity']
        G_copy = G.copy()
        G_copy[u][v]['capacity'] = 0
        flow_value = max_flow(G_copy, 's', 't')
        Li[(u, v)] = max(0, d - flow_value)
        Li[(v, u)] = Li[(u, v)]  # 对于无向边，保证两个方向的键一致
    return Li


# # 递归生成所有和为d的流向量组合_1
# def generate_flow_vectors(p, d):
#     def helper(p, d, current):
#         if p == 1:
#             yield current + [d]
#         else:
#             for i in range(d+1):
#                 yield from helper(p-1, d-i, current + [i])
#     return list(helper(p, d, []))

# # 枚举所有d-MP候选路径
# def enumerate_dMP_candidates(G, d, Li, paths):
#     p = len(paths)
#     Uj = [min([G[u][v]['capacity'] for u, v in zip(path[:-1], path[1:])]) for path in paths]
#     min_Uj = [min(uj, d) for uj in Uj]
#     candidates = []
#     begin_t = time.time()
#     flow_vectors = generate_flow_vectors(p, d)
#     print(len(flow_vectors))
#     end_t = time.time() - begin_t
#     print("生成流向量组合所需时间 = ", end_t)
#     begin_t = time.time()
#     for flow_vector in flow_vectors:
#         valid = True
#         x = {edge: 0 for edge in G.edges()}
#         for j, path in enumerate(paths):
#             if flow_vector[j] > min_Uj[j]:
#                 valid = False
#                 break
#             for u, v in zip(path[:-1], path[1:]):
#                 if (u, v) in x:
#                     x[(u, v)] += flow_vector[j]
#                 else:
#                     x[(v, u)] += flow_vector[j]  # 无向边，更新相反方向的边
#         for (u, v), capacity in x.items():
#             if capacity < Li[(u, v)] or capacity > min(G[u][v]['capacity'], d):
#                 valid = False
#                 break
#         if valid:
#             candidates.append(x)
#     end_t = time.time() - begin_t
#     print("验证流向量组合所需时间 = ", end_t)
#     return candidates


# 递归生成所有和为d的流向量组合_2
def generate_flow_vectors(p, d):
    def helper(p, d, current):
        if p == 1:
            yield current + [d]
        else:
            for i in range(d + 1):
                yield from helper(p - 1, d - i, current + [i])

    return list(helper(p, d, []))


# 构建路径-边矩阵
def build_path_edge_matrix(paths, edges):
    edge_index = {edge[:2]: i for i, edge in enumerate(edges)}
    p = len(paths)
    e = len(edges)
    A = np.zeros((p, e), dtype=int)
    for i, path in enumerate(paths):
        for u, v in zip(path[:-1], path[1:]):
            if (u, v) in edge_index:
                A[i][edge_index[(u, v)]] = 1
            else:
                A[i][edge_index[(v, u)]] = 1
    return A


# 枚举所有d-MP候选路径
def enumerate_dMP_candidates(G, d, Li, paths):
    edges = list(G.edges(data=True))
    p = len(paths)
    print("网络最小路数量 = ", p)
    Uj = np.array([min([G[u][v]['capacity'] for u, v in zip(path[:-1], path[1:])]) for path in paths])
    min_Uj = [min(uj, d) for uj in Uj]
    A = build_path_edge_matrix(paths, edges)

    candidates = []
    begin_t = time.time()
    flow_vectors = generate_flow_vectors(p, d)
    print("生成的流向量组合数 = ", len(flow_vectors))
    end_t = time.time() - begin_t
    print("生成流向量组合所需时间 = ", end_t)

    begin_t = time.time()
    for flow_vector in flow_vectors:
        flow_vector = np.array(flow_vector)
        if np.any(flow_vector > min_Uj):
            continue
        x = A.T @ flow_vector
        if all(Li[edge[:2]] <= x[i] <= min(G[edge[0]][edge[1]]['capacity'], d) for i, edge in enumerate(edges)):
            candidate = {edges[i][:2]: x[i] for i in range(len(edges))}
            candidates.append(candidate)

    end_t = time.time() - begin_t
    print("验证流向量组合所需时间 = ", end_t)
    return candidates


# 验证d-MP候选路径
def verify_dMP(G, candidate, d):
    # 复制图并根据候选路径更新边容量
    G_copy = G.copy()
    for edge, capacity in candidate.items():
        G_copy[edge[0]][edge[1]]['capacity'] = capacity
    # 按照论文中的步骤，验证d-MP候选路径
    for edge, capacity in candidate.items():
        u, v = edge
        if capacity > 0:
            G_copy[u][v]['capacity'] = capacity - 1
            if max_flow(G_copy, 's', 't') >= d:
                return False
            G_copy[u][v]['capacity'] = capacity  # 恢复边容量
    return True


# 删除重复的d-MP路径
def remove_duplicates(dMP_candidates):
    unique_candidates = []
    seen = set()
    for candidate in dMP_candidates:
        candidate_tuple = tuple(sorted(candidate.items()))
        if candidate_tuple not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate_tuple)
    return unique_candidates


# 将dMPs转换为矩阵形式
def convert_dMPs_to_matrix(dMPs, edge_index):
    matrix = []
    for dMP in dMPs:
        state_vector = [0] * len(edge_index)
        for edge, state in dMP.items():
            if edge in edge_index:
                state_vector[edge_index[edge]] = state
            # 处理相反方向的边
            else:
                state_vector[edge_index[(edge[1], edge[0])]] = state
        matrix.append(state_vector)
    return np.array(matrix)


# 主算法
def find_dMPs(G, d, edge_index):
    print("《算法开始》")
    Paths = list(nx.all_simple_paths(G, source='s', target='t'))
    start_time = time.time()
    Li = calculate_Li(G, d)
    dMP_candidates = enumerate_dMP_candidates(G, d, Li, Paths)
    begin_t = time.time()
    verified_dMPs = [candidate for candidate in dMP_candidates if verify_dMP(G, candidate, d)]
    end_t = time.time() - begin_t
    print("验证d-MPs所需时间 = ", end_t)
    begin_t = time.time()
    unique_dMPs = remove_duplicates(verified_dMPs)
    end_t = time.time() - begin_t
    print("d-MPs去重所需时间 = ", end_t)
    calcu_time = time.time() - start_time
    d_mp_matrix = convert_dMPs_to_matrix(unique_dMPs, edge_index)
    return d_mp_matrix, calcu_time


# 示例运行
G = read_network()
d = 3
# 用户指定的边编号顺序, 如果不需要编号则下面为默认选项
# edge_index = {edge[:2]: i for i, edge in enumerate(G.edges(data=True))}
# Example-1-NYF-2020-Fig-2（无向桥网络）
# edge_index = {('s', '1'): 0, ('1', 't'): 1, ('1', '2'): 2,
#               ('s', '2'): 3, ('2', 't'): 4}
# Example-2-许贝师姐大论文-ARPA网络（无向）-Fig.5.2-6节点9边
edge_index = {('s', '1'): 0, ('s', '2'): 1, ('1', '2'): 2,
              ('1', '3'): 3, ('2', '3'): 4, ('2', '4'): 5,
              ('3', '4'): 6, ('3', 't'): 7, ('4', 't'): 8}
d_MP_Mat, Calcu_time = find_dMPs(G, d, edge_index)
print("总计算时间 = ", Calcu_time)
# print(d_MP_Mat)
# np.save('Bridge_4_MP_Mat.npy', d_MP_Mat)
np.save('ARPA_3_MP_Mat.npy', d_MP_Mat)

