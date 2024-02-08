from .node import Node
from .treeoptimizer import TreeOptimizer

import copy
import random

def get_enable_subtree_dict(optimizer: TreeOptimizer):
    subtree_dict = optimizer.make_subtree_dict()
    tmp = dict()
    for K in subtree_dict.keys():
        to = TreeOptimizer()
        to.tree = subtree_dict[K]
        if not to.is_collapse():
            tmp[K] = subtree_dict[K]
    return tmp

def crossover(optimizer_A: TreeOptimizer, optimizer_B: TreeOptimizer):

    def crossover_func(optimizer_A: TreeOptimizer, optimizer_B: TreeOptimizer):
        enable_subtree_dict_A = get_enable_subtree_dict(optimizer_A)
        enable_subtree_dict_B = get_enable_subtree_dict(optimizer_B)

        tmp = list(enable_subtree_dict_A.keys())
        index_A = tmp[random.randint(0, len(tmp) - 1)]
        tmp = list(enable_subtree_dict_B.keys())
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
    
    tmp_optimizer_A, tmp_optimizer_B  = crossover_func(optimizer_A, optimizer_B)
    return tmp_optimizer_A, tmp_optimizer_B