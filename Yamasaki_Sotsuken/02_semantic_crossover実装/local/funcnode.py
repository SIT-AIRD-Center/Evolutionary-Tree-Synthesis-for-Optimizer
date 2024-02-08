from .node import Node
from .constnode import ConstNode
from .varnode import VarNode
from .func import *

import random
import tensorflow as tf

class FuncNode(Node):
    functions = [
            AddFunc(), SubFunc(), MulFunc(), DivFunc(),
            LogFunc(), Log1pFunc(),
            SinFunc(), CosFunc(), TanFunc(),SinhFunc(), CoshFunc(), TanhFunc(),
            ArcsinFunc(), ArccosFunc(), ArctanFunc(), ArcsinhFunc(), ArccoshFunc(), ArctanhFunc(),
            SqrtFunc(), SquareFunc(), PowFunc(), ExpFunc(), AbsFunc(), MaxFunc()
        ]

    functions_dict = {f"{func.__class__.__name__}": func for func in functions}

    # 引数が関数になる確率
    make_function_probability = 0.6
    # 引数が定数になる確率
    make_constant_probability = 0.1
    # 突然変異の確率
    mutate_probability = 0.1

    def __init__(self, max_depth: int, depth: int = 1, function: BaseFunc = None):
        super().__init__(depth = depth, node_type = "function")

        self.max_depth = max_depth

        if function is None:
            self.set_random_func()
        else:
            self.set_func(function)
        self.set_random_args()

        self.old_node_id = None
        self.save_flag = False
        self.calc_result = list()
    
    def set_random_func(self):
        index = random.randint(0, len(self.functions) - 1)
        self.function = self.functions[index]
    
    def set_func(self, function: BaseFunc):
        self.function = function
    
    def set_random_args(self):
        self.args = list()

        for _ in range(self.function.num_args):
            self.args.append( self.select_random_arg() )
    
    def select_random_arg(self):
        if self.depth + 2 > self.max_depth:
            if random.random() < 0.5:
                tmp = ConstNode(depth = self.depth + 1)
            else:
                tmp = VarNode(depth = self.depth + 1)
        
        else:
            p = random.random()
            if p < self.make_function_probability:
                tmp = FuncNode(max_depth = self.max_depth, depth = self.depth + 1)
            elif p < self.make_function_probability + self.make_constant_probability:
                tmp = ConstNode(depth = self.depth + 1)
            else:
                tmp = VarNode(depth = self.depth + 1)
        
        return tmp
    
    def reset_random_args(self):
        # 1 -> 2
        if len(self.args) < self.function.num_args:
            self.args.append( self.select_random_arg() )
        # 2 -> 1
        elif len(self.args) > self.function.num_args:
            if random.random() < 0.5:
                self.args = [self.args[0]]
            else:
                self.args = [self.args[1]]
    
    def set_save_flag(self, struct_dict: dict):
        self.save_flag = False
        for K in struct_dict.keys():
            if struct_dict[K]["node_type"] == "variable":
                if struct_dict[K]["variable_id"] == self.node_id:
                    self.save_flag = True
                    break
        
        for A in self.args:
            A.set_save_flag(struct_dict)
    
    def build(self, var_list: list):
        self.slots = list()
        if self.save_flag:
            for V in var_list:
                self.slots.append( tf.Variable(tf.zeros(V.shape)) )
        
        for A in self.args:
            A.build(var_list)
    
    def set_node_id(self, node_ids: list[int]):
        super().set_node_id(node_ids)

        for A in self.args:
            A.set_node_id(node_ids)
    
    def mutate_func(self):
        if random.random() < self.mutate_probability:
            self.set_random_func()
            self.reset_random_args()

        for A in self.args:
            A.mutate_func()

    def mutate_const(self):
        for A in self.args:
            A.mutate_const()

    def mutate_var(self, struct_dict: dict):
        for A in self.args:
            A.mutate_var(struct_dict)
    
    def get_struct(self, parent_id: int = None, is_second: bool = None):
        tmp = {
            "node_type" : self.node_type,
            "function" : self.function.__class__.__name__,
            "depth" : self.depth
        }
        if parent_id is not None: tmp["parent_id"] = parent_id
        if is_second is not None: tmp["is_second"] = is_second
        if self.old_node_id is not None: tmp["old_node_id"] = self.old_node_id
        return tmp
    
    def make_struct_dict(self, struct_dict: dict, parent_id: int = 0, is_second: bool = False):
        struct_dict[self.node_id] = self.get_struct(parent_id, is_second)
        
        for i, A in enumerate(self.args):
            if i == 0:
                A.make_struct_dict(struct_dict, parent_id = self.node_id)
            else:
                A.make_struct_dict(struct_dict, parent_id = self.node_id, is_second = True)

    def set_random_id(self, struct_dict: dict):
        for A in self.args:
            A.set_random_id(struct_dict)
    
    def reset_random_id(self, struct_dict: dict):
        for A in self.args:
            A.reset_random_id(struct_dict)

    def reset_node_id(self, node_ids: list[int]):
        if self.save_flag:
            self.old_node_id = self.node_id
        
        super().reset_node_id(node_ids)

        for A in self.args:
            A.reset_node_id(node_ids)

    def initialize_old_node_id(self):
        self.old_node_id = None

        for A in self.args:
            A.initialize_old_node_id()

    def get_subtree(self):
        return self
    
    def make_subtree_dict(self, subtree_dict: dict):
        subtree_dict[self.node_id] = self.get_subtree()

        for A in self.args:
            A.make_subtree_dict(subtree_dict)
    
    def get_variable(self, slot_index: int):
        return self.slots[slot_index]

    def make_variable_dict(self, variable_dict: dict, slot_index: int):
        if self.save_flag:
            variable_dict[self.node_id] = self.get_variable(slot_index)
        
        for A in self.args:
            A.make_variable_dict(variable_dict, slot_index)
    
    def __call__(self, variable_dict: dict, slot_index: int):
        args = list()
        for A in self.args:
            args.append( A(variable_dict = variable_dict, slot_index = slot_index) )


        result = self.function(args)
        
        if self.save_flag:
            self.slots[slot_index].assign( result )

        return result
    