from .node import Node
from .constnode import ConstNode
from .varnode import VarNode
from .funcnode import FuncNode
from .treeoptimizer import TreeOptimizer
from .func import *

class SGD(TreeOptimizer):
    def __init__(self, learning_rate: float = 0.001):

        self.model_params = list()
        self.grads_params = list()

        self.tree = FuncNode(max_depth = self.max_depth, function = MulFunc())
        self.tree.args[0] = ConstNode(depth = 2, constant = learning_rate)
        self.tree.args[1] = VarNode(depth = 2, variable_id = "gradient")

        self.set_node_id()
        self.set_save_flag()
        self.initialize_state()

class Momentum(TreeOptimizer):
    def __init__(self, learning_rate:float = 0.001, momentum=0.9):
        self.model_params = list()
        self.grads_params = list()
        self.tree = FuncNode(max_depth= self.max_depth, function = AddFunc())
        self.tree.args[0] = FuncNode(max_depth=self.max_depth, function=MulFunc())
        self.tree.args[0].args[0] = ConstNode(depth = 3, constant = momentum)
        self.tree.args[0].args[1] = VarNode(depth = 3, variable_id=1)
        self.tree.args[1] = FuncNode(max_depth=self.max_depth, function=MulFunc())
        self.tree.args[1].args[0] = ConstNode(depth=3, constant=learning_rate)
        self.tree.args[1].args[1] = VarNode(depth=1, variable_id="gradient")

        self.set_node_id()
        self.set_save_flag()
        self.initialize_state()

class RMSProp(TreeOptimizer):
    def __init__(self, learning_rate:float = 0.001, rho = 0.9, epsilon = 1e-7):
        self.model_params = list()
        self.grads_params = list()
        self.tree = FuncNode(max_depth= self.max_depth, function = MulFunc()) # 1
        self.tree.args[0] = FuncNode(max_depth= self.max_depth, function = DivFunc()) # 2
        self.tree.args[0].args[0] = ConstNode(depth = 3, constant = learning_rate) # 3
        self.tree.args[0].args[1] = FuncNode(max_depth= self.max_depth, function = SqrtFunc()) # 4
        self.tree.args[0].args[1].args[0] = FuncNode(max_depth= self.max_depth, function = AddFunc()) # 5
        self.tree.args[0].args[1].args[0].args[0] = FuncNode(max_depth= self.max_depth, function = AddFunc()) # 6
        self.tree.args[0].args[1].args[0].args[0].args[0] = FuncNode(max_depth= self.max_depth, function = MulFunc()) # 7
        self.tree.args[0].args[1].args[0].args[0].args[0].args[0] = ConstNode(depth = 7, constant = rho) # 8
        self.tree.args[0].args[1].args[0].args[0].args[0].args[1] = VarNode(depth = 7, variable_id = 6) # 9
        self.tree.args[0].args[1].args[0].args[0].args[1] = FuncNode(max_depth= self.max_depth, function = MulFunc()) # 10
        self.tree.args[0].args[1].args[0].args[0].args[1].args[0] = FuncNode(max_depth= self.max_depth, function = SubFunc()) # 11
        self.tree.args[0].args[1].args[0].args[0].args[1].args[0].args[0] = ConstNode(depth = 8, constant = 1) # 12
        self.tree.args[0].args[1].args[0].args[0].args[1].args[0].args[1] = ConstNode(depth = 8, constant = rho) # 13
        self.tree.args[0].args[1].args[0].args[0].args[1].args[1] = FuncNode(max_depth= self.max_depth, function = SquareFunc()) # 14
        self.tree.args[0].args[1].args[0].args[0].args[1].args[1].args[0] = VarNode(depth = 8, variable_id = "gradient") # 15
        self.tree.args[0].args[1].args[0].args[1] = ConstNode(depth = 5, constant = epsilon) # 16
        self.tree.args[1] = VarNode(depth = 2, variable_id = "gradient") # 17

        self.set_node_id()
        self.set_save_flag()
        self.initialize_state()