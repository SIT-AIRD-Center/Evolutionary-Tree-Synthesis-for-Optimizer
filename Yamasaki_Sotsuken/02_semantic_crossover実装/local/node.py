class Node:
    def __init__(self, node_type: str, depth: int = 1):

        # ノードの種類
        #   constant : 定数
        #   function : 関数
        #   variable : 変数
        self.node_type = node_type

        # ノードの深さ
        self.depth = depth

        # ノードのID
        self.node_id = None

    # ノードのIDを決定する関数
    #   node_ids : 既に使用されたノードのIDが格納されている配列
    def set_node_id(self, node_ids: list[int]):
        try:
            self.node_id = max(node_ids) + 1
        except:
            self.node_id = 1

        # 使用済みのIDを追加
        node_ids.append( self.node_id )
    
    # 突然変異・交叉した後にノードのIDを決定する関数
    def reset_node_id(self, node_ids: list[int]):
        try:
            self.node_id = max(node_ids) + 1
        except:
            self.node_id = 1
        
        # 使用済みのIDを追加
        node_ids.append( self.node_id )
    
    # old_node_idsを初期化する関数
    def initialize_old_node_id(self):
        pass
    
    def build(self, *args):
        pass
    
    def set_random_id(self, *args):
        pass

    def reset_random_id(self, *args):
        pass
    
    def set_save_flag(self, *args):
        pass
    
    def make_variable_dict(self, *args):
        pass

    def make_subtree_dict(self, *args):
        pass

    def mutate_const(self, *args):
        pass

    def mutate_func(self, *args):
        pass

    def mutate_var(self, *args):
        pass