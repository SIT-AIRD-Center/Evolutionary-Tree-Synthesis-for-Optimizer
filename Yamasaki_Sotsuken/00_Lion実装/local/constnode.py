from .node import Node

import random
import tensorflow as tf

class ConstNode(Node):

    # 突然変異する確率
    mutate_probability = 0.1

    def __init__(self, depth: int = 1, constant: float = None):
        super().__init__(depth = depth, node_type = "constant")

        if constant is None:
            self.set_random_constant()
        else:
            self.set_constant(constant)
    
    def set_random_constant(self):
        self.constant = random.normalvariate(0, 1)
    
    def set_constant(self, constant):
        self.constant = constant
    
    def build(self, var_list):
        self.slots = list()
        for V in var_list:
            self.slots.append( tf.Variable( tf.fill( V.shape, tf.cast(self.constant, tf.float32) ) )  )
 
    def __call__(self, variable_dict: dict, slot_index: int):
        return self.slots[slot_index]

    def mutate_const(self):
        if random.random() < self.mutate_probability:
            self.set_random_constant()
    
    def get_struct(self, parent_id: int = None, is_second: bool = None):
        tmp = {
            "node_type" : self.node_type,
            "constant" : self.constant,
            "depth" : self.depth,
        }
        if parent_id is not None: tmp["parent_id"] = parent_id
        if is_second is not None: tmp["is_second"] = is_second
        return tmp
    
    def make_struct_dict(self, struct_dict: dict, parent_id: int, is_second: bool = False):
        struct_dict[self.node_id] = self.get_struct(parent_id, is_second)