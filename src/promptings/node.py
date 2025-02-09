class Node:
    def __init__(self, task, plan=None, parent=None, depth=0, iters=0):
        self.task = task
        self.plan = plan
        self.parent = parent
        self.depth = depth
        self.iters = iters