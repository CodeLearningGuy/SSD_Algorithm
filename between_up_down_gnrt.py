import itertools
import numpy as np


def generate_vectors(lower_bound, upper_bound):
    # 为每个维度创建一个范围列表
    ranges = [range(lower, upper + 1) for lower, upper in zip(lower_bound, upper_bound)]

    # 使用 itertools.product 生成所有组合
    vectors = list(itertools.product(*ranges))

    # 转换为 numpy 数组以便更好地显示
    vectors_array = np.array(vectors)

    return vectors_array


def merge_and_remove_duplicates(matrix1, matrix2):
    # 将两个矩阵上下合并
    merged_matrix = np.vstack((matrix1, matrix2))

    # 使用 np.unique 去除重复的行
    unique_matrix = np.unique(merged_matrix, axis=0)

    return unique_matrix


def partial_order_sort(matrix):
    # 将矩阵转换为元组列表，方便比较
    tuple_list = [tuple(row) for row in matrix]

    # 自定义排序函数，按照指定规则进行比较
    def compare(row1, row2):
        for a, b in zip(row1, row2):
            if a < b:
                return -1
            elif a > b:
                return 1
        return 0

    # 使用 sorted 函数进行排序，key 使用自定义比较函数
    sorted_tuples = sorted(tuple_list, key=lambda x: [compare(x, y) for y in tuple_list])

    # 将排序后的元组列表转换回 NumPy 数组
    sorted_matrix = np.array(sorted_tuples)

    return sorted_matrix


def matrices_are_equal(matrix1, matrix2):
    # 首先判断形状是否相同
    if matrix1.shape != matrix2.shape:
        return False

    # 判断元素是否相同
    return np.array_equal(matrix1, matrix2)


# # 示例
# lower_bound = [1, 2, 1, 2, 1]
# upper_bound = [3, 2, 3, 3, 3]
# result_1 = generate_vectors(lower_bound, upper_bound)
#
# lower_bound_2 = [1, 2, 1, 1, 2]
# upper_bound_2 = [3, 2, 3, 1, 3]
# result_2 = generate_vectors(lower_bound_2, upper_bound_2)
#
# lower_bound_3 = [1, 2, 1, 1, 2]
# upper_bound_3 = [3, 2, 3, 3, 3]
# result_3 = generate_vectors(lower_bound_3, upper_bound_3)
#
# result_4 = merge_and_remove_duplicates(result_1, result_3)
#
# result_5 = np.vstack((result_1, result_2))
#
# sorted_result_4 = partial_order_sort(result_4)
# sorted_result_5 = partial_order_sort(result_5)
# print(matrices_are_equal(sorted_result_4, sorted_result_5))

lower_bound = [2, 1, 1, 1, 2]
upper_bound = [3, 3, 2, 3, 3]
lower_bound_2 = [2, 1, 1, 2, 1]
upper_bound_2 = [3, 3, 2, 3, 1]
result_1 = generate_vectors(lower_bound, upper_bound)
result_2 = generate_vectors(lower_bound_2, upper_bound_2)

lower_bound_3 = [2, 1, 1, 2, 1]
upper_bound_3 = [3, 3, 2, 3, 3]
lower_bound_4 = [2, 1, 1, 1, 2]
upper_bound_4 = [3, 3, 2, 1, 3]
result_3 = generate_vectors(lower_bound_3, upper_bound_3)
result_4 = generate_vectors(lower_bound_4, upper_bound_4)
# lower_bound_5 = [2, 1, 1, 1, 2]
# upper_bound_5 = [3, 3, 2, 3, 3]
# lower_bound_6 = [2, 1, 3, 1, 2]
# upper_bound_6 = [3, 3, 3, 1, 2]

# result_1 = generate_vectors(lower_bound, upper_bound)
# result_2 = generate_vectors(lower_bound_2, upper_bound_2)
# result_3 = generate_vectors(lower_bound_3, upper_bound_3)
# result_4 = generate_vectors(lower_bound_4, upper_bound_4)
# result_5 = generate_vectors(lower_bound_5, upper_bound_5)
# result_6 = generate_vectors(lower_bound_6, upper_bound_6)
Result_1 = np.vstack((result_1, result_2))

# lower_bound_7 = [2, 1, 1, 2, 1]
# upper_bound_7 = [3, 3, 2, 3, 3]
# lower_bound_8 = [2, 1, 3, 2, 1]
# upper_bound_8 = [3, 3, 3, 2, 2]
# lower_bound_9 = [2, 1, 1, 1, 2]
# upper_bound_9 = [3, 3, 2, 1, 3]
# lower_bound_10 = [2, 1, 3, 1, 2]
# upper_bound_10 = [3, 3, 3, 1, 2]
# result_7 = generate_vectors(lower_bound_7, upper_bound_7)
# result_8 = generate_vectors(lower_bound_8, upper_bound_8)
# result_9 = generate_vectors(lower_bound_9, upper_bound_9)
# result_10 = generate_vectors(lower_bound_10, upper_bound_10)
# Result_4 = merge_and_remove_duplicates(result_7, merge_and_remove_duplicates(result_8, result_9))
# Result_5 = merge_and_remove_duplicates(Result_4, result_10)
Result_2 = np.vstack((result_3, result_4))

sorted_result_1 = partial_order_sort(Result_1)
sorted_result_2 = partial_order_sort(Result_2)
print(matrices_are_equal(sorted_result_1, sorted_result_2))

