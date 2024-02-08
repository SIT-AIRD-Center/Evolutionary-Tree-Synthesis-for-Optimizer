from .node import Node

import random

class VarNode(Node):
    # 変数が勾配かパラメータになる確率
    gradient_or_parametor_probability = 0.2

    def __init__(self, depth: int = 1, variable_id = None):
        super().__init__(depth = depth, node_type = "variable")

        if variable_id is not None:
            self.set_variable_id(variable_id)
        else:
            self.variable_id = None
    
    def set_random_id(self, struct_dict: dict):
        if random.random() < self.gradient_or_parametor_probability:
            if random.random() < 0.5:
                self.variable_id = "gradient"
            else:
                self.variable_id = "parameter"
        
        else:
            struct_dict_keys = list(struct_dict.keys())
            index = random.randint(0, len(struct_dict_keys) - 1)
            while struct_dict[struct_dict_keys[index]]["node_type"] != "function":
                index = random.randint(0, len(struct_dict_keys) - 1)
            self.set_variable_id(struct_dict_keys[index])
    
    def set_variable_id(self, variable_id):
        self.variable_id = variable_id
    
    def reset_random_id(self, struct_dict: dict):
        flag = True
        if self.variable_id in ["gradient", "parameter", "iteration"]:
            flag = False
        else:
            for K in struct_dict.keys():
                try:
                    if struct_dict[K]["old_node_id"] == self.variable_id:
                        self.variable_id = K
                        flag = False
                        break
                except:
                    pass
        
        if flag: self.set_random_id(struct_dict)
    
    def __call__(self, variable_dict: dict, slot_index: int):
        return variable_dict[self.variable_id]

    def mutate_var(self, struct_dict: dict):
        self.reset_random_id(struct_dict)

    def get_struct(self, parent_id: int = None, is_second: bool = None):
        tmp = {
            "node_type" : self.node_type,
            "variable_id" : self.variable_id,
            "depth" : self.depth
        }
        if parent_id is not None: tmp["parent_id"] = parent_id
        if is_second is not None: tmp["is_second"] = is_second
        return tmp
    
    def make_struct_dict(self, struct_dict: dict, parent_id: int, is_second: bool = False):
        struct_dict[self.node_id] = self.get_struct(parent_id, is_second)