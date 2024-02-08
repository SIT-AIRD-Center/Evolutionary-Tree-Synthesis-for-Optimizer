from .node import Node
from .treeoptimizer import TreeOptimizer

import copy
import random
import numpy as np
import tensorflow as tf

def get_enable_subtree_dict(optimizer: TreeOptimizer):
    subtree_dict = optimizer.make_subtree_dict()
    tmp = dict()
    for K in subtree_dict.keys():
        to = TreeOptimizer()
        to.tree = subtree_dict[K]
        if not to.is_collapse():
            tmp[K] = subtree_dict[K]
    return tmp

def crossover(optimizer_A: TreeOptimizer, optimizer_B: TreeOptimizer, index_A = None, index_B = None):

    enable_subtree_dict_A = get_enable_subtree_dict(optimizer_A)
    enable_subtree_dict_B = get_enable_subtree_dict(optimizer_B)

    tmp = list(enable_subtree_dict_A.keys())
    if index_A == None:
        index_A = tmp[random.randint(0, len(tmp) - 1)]
    tmp = list(enable_subtree_dict_B.keys())
    if index_B == None:
        index_B = tmp[random.randint(0, len(tmp) - 1)]

    struct_dict_A = optimizer_A.make_struct_dict() 

    optimizer_A = copy.deepcopy(optimizer_A)
    if struct_dict_A[index_A]["parent_id"] == 0:
        optimizer_A.tree = copy.deepcopy( enable_subtree_dict_B[index_B] )
    else:
        tmp_subtree_dict = optimizer_A.make_subtree_dict()
        if struct_dict_A[index_A]["is_second"] == False:
            tmp_subtree_dict[ struct_dict_A[index_A]["parent_id"] ].args[0] = copy.deepcopy( enable_subtree_dict_B[index_B] )
        else:
            tmp_subtree_dict[ struct_dict_A[index_A]["parent_id"] ].args[1] = copy.deepcopy( enable_subtree_dict_B[index_B] )
    
    optimizer_A.reset_node_id()
    optimizer_A.reset_random_id()
    optimizer_A.initialize_state()
    optimizer_A.tree.initialize_old_node_id()

    struct_dict_B = optimizer_B.make_struct_dict() 

    optimizer_B = copy.deepcopy(optimizer_B)
    if struct_dict_B[index_B]["parent_id"] == 0:
        optimizer_B.tree = copy.deepcopy( enable_subtree_dict_A[index_A] )
    else:
        tmp_subtree_dict = optimizer_B.make_subtree_dict()
        if struct_dict_B[index_B]["is_second"] == False:
            tmp_subtree_dict[ struct_dict_B[index_B]["parent_id"] ].args[0] = copy.deepcopy( enable_subtree_dict_A[index_A] )
        else:
            tmp_subtree_dict[ struct_dict_B[index_B]["parent_id"] ].args[1] = copy.deepcopy( enable_subtree_dict_A[index_A] )
    
    optimizer_B.reset_node_id()
    optimizer_B.reset_random_id()
    optimizer_B.initialize_state()
    optimizer_B.tree.initialize_old_node_id()

    return optimizer_A, optimizer_B

def euclidean_distance(input_a, input_b):
    output = 0.0
    for a, b in zip(input_a, input_b):
        for c, d in zip(a, b):
            output += np.sum(np.square(np.subtract(c, d)))
    # ゼロ除算対策
    return np.sqrt(output + 1e-5)

def semantic_crossover(parent1, parent2, t, G, a_min = -2, a_max = 2):
    tmp = TreeOptimizer()
    # parent1からランダムに部分木を決める
    enable_subtree_dict_A = get_enable_subtree_dict(parent1)
    index_A = random.choice(list(enable_subtree_dict_A.keys()))
    parent1_sub_tree = enable_subtree_dict_A[index_A]
    # 決めた部分木の出力例を求める
    parent1_sub_tree_outputs = list()
    tmp.tree = parent1_sub_tree
    for params, grads in zip(parent1.model_params, parent1.grads_params):
        parent1_sub_tree_outputs.append(tmp.calc_tree(zip(params, grads)))

    # parent2から列挙できる部分木を全て求める
    enable_subtree_dict_B = get_enable_subtree_dict(parent2)
    # 列挙した各部分木の出力例を求める
    parent2_sub_tree_outputs = dict()
    for key in enable_subtree_dict_B.keys():
        tmp.tree = enable_subtree_dict_B[key]
        output = list()
        for params, grads in zip(parent2.model_params, parent2.grads_params):
            output.append(tmp.calc_tree(zip(params, grads)))
        parent2_sub_tree_outputs[key] = output

    distance_list = list()
    distance_keys = list()
    # 類似度を求める
    for key in parent2_sub_tree_outputs.keys():
        distance_list.append(euclidean_distance(parent1_sub_tree_outputs, parent2_sub_tree_outputs[key]))
        distance_keys.append(key)

    alpha = a_max + (a_min - a_max) * t / G
    distance_list = np.power(distance_list, alpha)

    for i in range(len(distance_list)):
        if np.isnan(distance_list[i]) or np.isinf(distance_list[i]):
            distance_list[i] = 1e10
        elif distance_list[i] == 0.0:
            distance_list[i] = 0.01
    # 確率に変換 numpy関数で計算しないと合計1.0にならない判定をもらう
    p = np.divide(distance_list, np.sum(distance_list))

    # 部分木の決定
    index_B = np.random.choice(distance_keys,p=p)
    # parent2_sub_tree = enable_subtree_dict_B[index_B]

    return crossover(parent1, parent2, index_A, index_B)