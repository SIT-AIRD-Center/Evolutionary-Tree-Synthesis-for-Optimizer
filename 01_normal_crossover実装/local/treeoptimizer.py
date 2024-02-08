from .node import Node
from .constnode import ConstNode
from .varnode import VarNode
from .funcnode import FuncNode

import copy
import random
import graphviz
import tensorflow as tf

class TreeOptimizer:

    mutate_probability = 0.1
    max_depth = 5

    def __init__(self):

        self.tree = FuncNode(max_depth = self.max_depth)
        self.set_node_id()
        self.set_random_id()

        # 木が勾配を含んでいるか確認
        while not self.has_gradient():
            self.tree = FuncNode(max_depth = self.max_depth)
            self.set_node_id()
            self.set_random_id()

        # self.set_save_flag()
        self.initialize_state()

    def set_node_id(self):
        node_ids = list()
        self.tree.set_node_id(node_ids)

    def reset_node_id(self):
        node_ids = list()
        self.tree.reset_node_id(node_ids)

    def set_random_id(self):
        struct_dict = self.make_struct_dict()
        self.tree.set_random_id(struct_dict)
    
    def reset_random_id(self):
        struct_dict = self.make_struct_dict()
        self.tree.reset_random_id(struct_dict)
    
    def set_save_flag(self):
        struct_dict = self.make_struct_dict()
        self.tree.set_save_flag(struct_dict)

    def initialize_state(self):
        self.is_built = False
        self.iteration = 0.0
    
    def has_gradient(self):
        struct_dict = self.make_struct_dict()
        for K in struct_dict.keys():
            if struct_dict[K]["node_type"] == "variable":
                if struct_dict[K]["variable_id"] == "gradient":
                    return True
        return False
    
    def build(self, var_list):
        self.set_save_flag()
        self.tree.build(var_list)
        self.iteration_slots = list()
        for V in var_list:
            self.iteration_slots.append( tf.Variable( tf.zeros(V.shape) ) )
        self.is_built = True
    
    def make_struct_dict(self):
        struct_dict = dict()
        self.tree.make_struct_dict(struct_dict)
        return struct_dict

    def make_variable_dict(self, slot_index: int):
        variable_dict = dict()
        self.tree.make_variable_dict(variable_dict, slot_index)
        return variable_dict
    
    def make_subtree_dict(self):
        subtree_dict = dict()
        self.tree.make_subtree_dict(subtree_dict)
        return subtree_dict

    def apply_gradients(self, grad_and_vars):
        if self.is_built == False:
            var_list = [GaV[1] for GaV in grad_and_vars]
            self.build(var_list)
        
        for i, (G, V) in enumerate(grad_and_vars):
            self.update_step(G, V, i)
        
    
    def calc_tree(self, grad_and_vars):
        if self.is_built == False:
            var_list = [GaV[1] for GaV in grad_and_vars]
            self.build(var_list)
        output = []
        for i, (G, V) in enumerate(grad_and_vars):
        
            output.append(self.calc_tree_output(G, V, i))
        
        return output
        
    
    def mutate(self):

        while random.random() < self.mutate_probability:
            self.mutate_node()

        self.tree.mutate_const()

        self.mutate_func()

        self.mutate_var()

        if not self.has_gradient():
            gradient_candidate_keys = list()

            struct_dict = self.make_struct_dict()
            for key in sorted(struct_dict.keys()):
                if struct_dict[key]["node_type"] != "function":
                    gradient_candidate_keys.append(key)
            
            select_key = random.choice(gradient_candidate_keys)

            struct_dict[select_key]["node_type"] = "variable"
            struct_dict[select_key]["variable_id"] = "gradient"

            self.tree = self.create_tree_from_dict(struct_dict)

        self.set_node_id()
        self.initialize_state()
    
    def mutate_var(self):

        def mutate_var_func():
            tmp_optimizer = copy.deepcopy(self)
            struct_dict = tmp_optimizer.make_struct_dict()
            tmp_optimizer.tree.mutate_var(struct_dict)
            return tmp_optimizer

        tmp_optimizer = mutate_var_func()
        # while not tmp_optimizer.has_gradient():
        #     tmp_optimizer = mutate_var_func()
        
        self.tree = tmp_optimizer.tree
    
    def mutate_func(self):

        def mutate_func_func():
            tmp_optimizer = copy.deepcopy(self)
            tmp_optimizer.tree.mutate_func()

            self.set_node_id()

            return tmp_optimizer

        tmp_optimizer = mutate_func_func()
        # while not tmp_optimizer.has_gradient():
        #     tmp_optimizer = mutate_func_func()
        
        self.tree = tmp_optimizer.tree
        self.set_node_id()

    def mutate_node(self):

        def mutate_node_func():
            tmp_optimizer = copy.deepcopy(self)

            tmp_optimizer = copy.deepcopy(self)
            struct_dict = tmp_optimizer.make_struct_dict()
            struct_dict_keys = list(struct_dict.keys())
            index = random.randint(0, len(struct_dict_keys) - 1)
            subtree_dict = tmp_optimizer.make_subtree_dict()

            parent_id = struct_dict[struct_dict_keys[index]]["parent_id"]

            # ルートノードを作り直す
            if parent_id == 0:
                tmp_optimizer.tree = FuncNode(max_depth = tmp_optimizer.max_depth)
            # 親ノードの引数からノードを選択
            else:
                if struct_dict[struct_dict_keys[index]]["is_second"] == False:
                    subtree_dict[parent_id].args[0] = tmp_optimizer.make_random_node(depth = struct_dict[parent_id]["depth"])
                else:
                    subtree_dict[parent_id].args[1] = tmp_optimizer.make_random_node(depth = struct_dict[parent_id]["depth"])
            
            # idを振りなおす
            tmp_optimizer.reset_node_id()
            tmp_optimizer.reset_random_id()
            tmp_optimizer.tree.initialize_old_node_id()

            return tmp_optimizer

        tmp_optimizer = mutate_node_func()
        # while not tmp_optimizer.has_gradient():
        #     tmp_optimizer = mutate_node_func()
        
        self.tree = tmp_optimizer.tree

    def make_random_node(self, depth: int):
        if depth + 2 > self.max_depth:
            if random.random() < 0.5:
                return ConstNode(depth = depth + 1)
            else:
                return VarNode(depth = depth + 1)
        
        else:
            p = random.random()
            if p < 0.33:
                return ConstNode(depth = depth + 1)
            elif p < 0.66:
                return VarNode(depth = depth + 1)
            else:
                return FuncNode(max_depth = self.max_depth, depth = depth + 1)

    
    def update_step(self, gradient, parameter, slot_index):
        variable_dict = self.make_variable_dict(slot_index)
        variable_dict["gradient"] = gradient
        variable_dict["parameter"] = parameter
        self.iteration_slots[slot_index].assign_add(tf.ones(parameter.shape))
        variable_dict["iteration"] = self.iteration_slots[slot_index]

        result = self.tree(variable_dict, slot_index)
        parameter.assign_sub(result)
    
    def calc_tree_output(self, gradient, parameter, slot_index):
        variable_dict = self.make_variable_dict(slot_index)
        variable_dict["gradient"] = gradient
        variable_dict["parameter"] = parameter
        self.iteration_slots[slot_index].assign_add(tf.ones(parameter.shape))
        variable_dict["iteration"] = self.iteration_slots[slot_index]

        result = self.tree(variable_dict, slot_index)

        return result        

    def plot_struct(self, file_name= None):
        struct_dict = self.make_struct_dict()

        dot = graphviz.Digraph()

        for K in struct_dict.keys():
            if struct_dict[K]["node_type"] == "function":
                node_name = f"function_{K} : {struct_dict[K]['function']}"
                dot.node(str(K), node_name)
            elif struct_dict[K]["node_type"] == "constant":
                node_name = f"constant_{K} : {struct_dict[K]['constant']}"
                dot.node(str(K), node_name)
            else:
                node_name = f"variable_{K} : {struct_dict[K]['variable_id']}"
                dot.node(str(K), node_name)
        
        for K in struct_dict.keys():
            if struct_dict[K]["parent_id"] == 0: continue

            dot.edge(str(struct_dict[K]['parent_id']), str(K), color = "blue" if struct_dict[K]['is_second'] else "black")

            if struct_dict[K]["node_type"] == "variable":
                if struct_dict[K]["variable_id"] not in ["gradient", "parameter", "iteration"]:
                    dot.edge(str(struct_dict[K]['variable_id']), str(K), color = "red", style = "dashed")
        if file_name == None:
            dot.render("optimizer", format = "png")
        else:
            dot.render(file_name, format = "png")

    def is_collapse(self):
        struct_dict = self.make_struct_dict()
        variable_ids = list()
        for K in struct_dict.keys():
            if struct_dict[K]["node_type"] == "variable":
                if struct_dict[K]["variable_id"] not in ["gradient", "parameter", "iteration"]:
                    variable_ids.append(struct_dict[K]["variable_id"])
        
        flag = False
        for I in variable_ids:
            try:
                struct_dict[I]
            except:
                flag = True
        return flag
    
    # 辞書から木構造を作成する関数
    def create_tree_from_dict(self, tree_dict):
        tmp_dict = {}
        relation_dict = {}
        tmp = TreeOptimizer()
        for key in sorted(tree_dict.keys(), reverse=1):
            if not tree_dict[key]["parent_id"] in relation_dict.keys():
                relation_dict[tree_dict[key]["parent_id"]] = []
            relation_dict[tree_dict[key]["parent_id"]].append(key)
            relation_dict[tree_dict[key]["parent_id"]] = sorted(relation_dict[tree_dict[key]["parent_id"]])
            
            if tree_dict[key]["node_type"] == "variable":
                tree = VarNode(depth = tree_dict[key]["depth"], variable_id = tree_dict[key]["variable_id"])
            elif tree_dict[key]["node_type"] == "constant":
                tree = ConstNode(depth = 0, constant = tree_dict[key]["constant"])
            else:
                tree = FuncNode(max_depth= tmp.max_depth, function = FuncNode.functions_dict[tree_dict[key]["function"]])
                for i, k in enumerate(relation_dict[key]):
                    tree.args[i] = tmp_dict[k]
            
            tmp_dict[key] = tree
        return tmp_dict[sorted(list(tmp_dict.keys()))[0]]