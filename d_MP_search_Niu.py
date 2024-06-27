

import networkx as nx
import matplotlib.pyplot as plt
import time
from networkx.algorithms.flow import preflow_push


# 读取网络数据
def read_network():
    G = nx.Graph()
    # # Example-1-桥网络（无向）
    # G.add_edge('s', '1', capacity=3)
    # G.add_edge('s', '2', capacity=2)
    # G.add_edge('1', '2', capacity=1)
    # G.add_edge('1', 't', capacity=2)
    # G.add_edge('2', 't', capacity=2)
    # Example-2-许贝师姐大论文-ARPA网络（无向）-Fig.5.2-6节点9边
    # G.add_edge('s', '1', capacity=7)
    # G.add_edge('s', '2', capacity=5)
    # G.add_edge('1', '2', capacity=4)
    # G.add_edge('1', '3', capacity=6)
    # G.add_edge('2', '3', capacity=4)
    # G.add_edge('2', '4', capacity=5)
    # G.add_edge('3', '4', capacity=4)
    # G.add_edge('3', 't', capacity=6)
    # G.add_edge('4', 't', capacity=7)
    G.add_edge('s', '1', capacity=10)
    G.add_edge('s', '2', capacity=10)
    G.add_edge('1', '2', capacity=10)
    G.add_edge('1', '3', capacity=10)
    G.add_edge('2', '3', capacity=10)
    G.add_edge('2', '4', capacity=10)
    G.add_edge('3', '4', capacity=10)
    G.add_edge('3', 't', capacity=10)
    G.add_edge('4', 't', capacity=10)
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


# # 根据定理2实现d-MP候选的枚举
# def enumerate_dMP_candidates(G, d, Li):
#     # 获取从源节点到目标节点的所有简单路径
#     paths = list(nx.all_simple_paths(G, source='s', target='t'))
#     # 计算路径的数量
#     p = len(paths)
#     # 计算每条路径Pj的最大容量Uj，Uj是每条路径上所有边的最小容量
#     Uj = [min([G[u][v]['capacity'] for u, v in zip(path[:-1], path[1:])]) for path in paths]
#     # 初始化d-MP候选路径列表
#     candidates = []
#     # 生成所有可能的流向量组合
#     for flow_vector in product(range(d + 1), repeat=p):
#         # 如果流向量的总和不等于d，则跳过该组合
#         if sum(flow_vector) != d:
#             continue
#         # 假设当前流向量组合有效
#         valid = True
#         # 初始化每条边的流量为0
#         x = {edge: 0 for edge in G.edges()}
#         # 遍历每条路径，检查路径上的流量是否符合约束
#         for j, path in enumerate(paths):
#             # 如果流量超过路径的最大容量，则标记为无效
#             if flow_vector[j] > Uj[j]:
#                 valid = False
#                 break
#             if flow_vector[j] != 0:
#                 # 更新路径上每条边的流量
#                 for u, v in zip(path[:-1], path[1:]):
#                     if (u, v) in x:
#                         x[(u, v)] += flow_vector[j]
#                     else:
#                         x[(v, u)] += flow_vector[j]  # 无向边，更新相反方向的边
#         if valid:
#             # 检查每条边的流量是否符合Li的限制
#             for (u, v), capacity in x.items():
#                 # 如果边的流量小于Li值或大于边的最大容量，则标记为无效
#                 if capacity < Li[(u, v)] or capacity > min(G[u][v]['capacity'], d):
#                     valid = False
#                     break
#         else:
#             continue
#         # 如果组合有效，则将其添加到候选列表中
#         if valid:
#             candidates.append(x)
#     # 返回所有满足条件的d-MP候选路径
#     return candidates


# 递归生成所有和为d的流向量组合_2
def generate_flow_vectors(p, d):
    def helper(p, d, current):
        if p == 1:
            yield current + [d]
        else:
            for i in range(d+1):
                yield from helper(p-1, d-i, current + [i])
    return list(helper(p, d, []))


# 枚举所有d-MP候选路径
def enumerate_dMP_candidates(G, d, Li, paths):
    p = len(paths)
    Uj = [min([G[u][v]['capacity'] for u, v in zip(path[:-1], path[1:])]) for path in paths]
    min_Uj = [min(uj, d) for uj in Uj]
    candidates = []
    begin_t = time.time()
    flow_vectors = generate_flow_vectors(p, d)
    print(len(flow_vectors))
    end_t = time.time() - begin_t
    print("生成流向量组合所需时间 = ", end_t)
    begin_t = time.time()
    for flow_vector in flow_vectors:
        valid = True
        x = {edge: 0 for edge in G.edges()}
        for j, path in enumerate(paths):
            if flow_vector[j] > min_Uj[j]:
                valid = False
                break
            for u, v in zip(path[:-1], path[1:]):
                if (u, v) in x:
                    x[(u, v)] += flow_vector[j]
                else:
                    x[(v, u)] += flow_vector[j]  # 无向边，更新相反方向的边
        for (u, v), capacity in x.items():
            if capacity < Li[(u, v)] or capacity > min(G[u][v]['capacity'], d):
                valid = False
                break
        if valid:
            candidates.append(x)
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

# 主算法
def find_dMPs(G, d):
    print("算法开始")
    Paths = list(nx.all_simple_paths(G, source='s', target='t'))
    start_time = time.time()
    Li = calculate_Li(G, d)
    dMP_candidates = enumerate_dMP_candidates(G, d, Li, Paths)
    begin_t = time.time()
    verified_dMPs = [candidate for candidate in dMP_candidates if verify_dMP(G, candidate, d)]
    end_t = time.time() - begin_t
    print("验证d-MPs所需时间 = ", end_t)
    unique_dMPs = remove_duplicates(verified_dMPs)
    calcu_time = time.time() - start_time
    return unique_dMPs, calcu_time

# 示例运行
G = read_network()
d = 20
dMPs, Calcu_time = find_dMPs(G, d)
# print('\n', dMPs)
print("计算时间 = ", Calcu_time)

