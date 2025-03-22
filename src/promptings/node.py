class Node:
    def __init__(self, plan, score=0):
        self.plan = plan
        self.code = ""
        self.children = []
        self.score = score
        self.depth = 0
        
    def sort_children(self):
        self.children.sort(key=lambda x:x.score, reverse=True)

    def set_code(self, code):
        self.code = code
    
    def set_depth(self, depth):
        self.depth = depth
        