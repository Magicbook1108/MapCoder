class Node:
    def __init__(self, plan, score):
        self.plan = plan
        self.code: str
        self.children = []
        self.score = score
        self.visits = 0
        self.depth = 0
        
    def sort_children(self,):
        self.children.sort(key=lambda x:x.score, reverse=True)

    def set_code(self, code):
        self.code = code