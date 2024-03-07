import numpy as np
import utils
import torch

VERBOSE = False
SHOW_PREDICTIONS = False
total_node_expansion = 0
iteration_node_expansion = 0
MAX_NODE_EXPANSIONS = 1250000

class Node:

    def __init__(self, support_x, support_y, current_grid, parent_primitive, all_primitives, name="root", device='cuda'):

        self.parent_primitive = parent_primitive
        self.support_x = support_x
        self.support_y = support_y
        self.current_grid = current_grid
        self.all_primitives = all_primitives
        self.name = name
        self.device = device

        self.h = None
        self.children = []

    def apply_primitive(self, prim_func, grid):
        return prim_func(grid)

    def successors(self, model):
        global total_node_expansion
        global iteration_node_expansion

        if len(self.children) == 0:
            total_node_expansion += len(self.all_primitives)
            iteration_node_expansion += len(self.all_primitives)

            tmp_children = []

            for idx in range(len(self.all_primitives)):

                prim = self.all_primitives[idx]
                prim_func = prim[0]
                transformed_grid = self.apply_primitive(prim_func, self.current_grid)

                child_node = Node(self.support_x, self.support_y,   # TODO: copies of the support set in each node is
                                                                    #  redundant
                                  transformed_grid, idx,
                                  self.all_primitives,              # TODO: also redundant?
                                  name="%s/%i" % (self.name, prim[1]))

                h_value = child_node.calc_h(model)
                tmp_children.append([h_value, child_node])

            # sort children based on their h value
            children = sorted(tmp_children, key=lambda x: int(x[0]))

            child_nodes = []
            for c in children:
                child_nodes.append(c[1])

            self.children = child_nodes

        return self.children

    def calc_h(self, model):
        if model is None:
            return 1.

        if self.h is None:
            self.h = model.evaluate(self.current_grid, self.support_y)
        else:
            return self.h

    # for the "in" operation
    def __eq__(self, other_node):
        return np.all(self.current_grid == other_node.current_grid)

class AStarSearch():

    @staticmethod
    def plan(root, model):
        global total_node_expansion
        global iteration_node_expansion

        total_node_expansion = 0
        iteration_node_expansion = 0

        bound = 0
        path = [root]

        found = False
        while not found and total_node_expansion < MAX_NODE_EXPANSIONS:
            found = AStarSearch.search(path, 0, bound, model)
            bound += 1
            iteration_node_expansion = 0

        action_seq = []
        for node_idx in range(1, len(path)):
            action_seq.append(path[node_idx].parent_action + 1)

        return action_seq, total_node_expansion

    MAX_G = 45

    @staticmethod
    def search(path, g, bound, model):

        path_txt = "[ROOT"
        if len(path) > 1:
            for node_idx in range(1, len(path)):
                path_txt += ",%i" % path[node_idx].parent_action

        path_txt += "]"

        node = path[-1]
        tmp_h = 0
        if len(path) > 1:
            tmp_h = node.calc_h(model)

        f_cost = g + tmp_h

        if g >= AStarSearch.MAX_G:
            if VERBOSE:
                print("==> Reached maximum depth %i. Aborting this path." % AStarSearch.MAX_G)
            return False

        if f_cost > bound:
            if VERBOSE:
                print("==> Cost exceeded bound, returning...")
            return False

        if len(path) > 1:
            if node.calc_h(model) == 0:
                print("===> FOUND GOAL after expanding %i nodes" % total_node_expansion)
                return True

        if total_node_expansion >= MAX_NODE_EXPANSIONS:
            print("===> Reached total_node_expansion of %i, returning..." % MAX_NODE_EXPANSIONS)
            return False

        found = False
        successors = node.successors()

        children = []
        for c in node.children:
            children.append(c.parent_action)

        for succ in successors:
            if succ not in path:
                path.append(succ)
                found = AStarSearch.search(path, g + 1, bound)
                if found:
                    return True

                path.pop()

        return found