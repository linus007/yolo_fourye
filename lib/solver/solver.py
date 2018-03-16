"""
Solver abstract class
"""
class Solver(object):
    def __init__(self, dataset, net, common_conf, solver_conf):
        raise NotImplementedError

    def solve(self):
        raise NotImplementedError
