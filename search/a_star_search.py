import numpy as np

VERBOSE = False
total_node_expansion = 0
iteration_node_expansion = 0
MAX_NODE_EXPANSIONS = 50000
HEURISTIC_TYPE = 'distance' # alternative: 'similarity'

class Node:

    def __init__(self, support_x, support_y, current_grid, parent_primitive, all_primitives, name="root", device='cuda'):

        self.parent_primitive = parent_primitive
        self.support_x = support_x
        self.support_y = support_y
        self.all_primitives = all_primitives
        self.name = name
        self.device = device
        self.current_grid = current_grid

        self.h = None
        self.children = []

    @staticmethod
    def apply_primitive(prim_func, grid_batch):
        transformed_batch = []
        for grid in grid_batch:
            tmp_grid = prim_func(grid)
            transformed_batch.append(tmp_grid)

        return transformed_batch

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
                                  transformed_grid, prim,
                                  self.all_primitives,              # TODO: also redundant?
                                  name="%s/%s" % (self.name, prim[1]))

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
        a = np.reshape(self.current_grid, [-1])
        b = np.reshape(self.support_y, [-1])
        found = np.all(a == b)

        if found:
            return 0
        else:
            if model is None:
                return 1

            if self.h is None:
                # h must be the distance to the goal, while the model returns the similarity. Hence the 1 - similarity to
                # get the value of h.
                pred = model.evaluate(np.array(self.current_grid), self.support_y)

                if HEURISTIC_TYPE == 'distance':
                    self.h = pred
                else:
                    self.h = int((1. - pred) * 100)

            return self.h

    # for the "in" operation
    def __eq__(self, other_node):
        a = np.reshape(self.current_grid, [-1])
        b = np.reshape(other_node.current_grid, [-1])
        return np.all(a == b)

    def __repr__(self):
        if self.parent_primitive is None:
            return 'root'

        return self.parent_primitive[1]

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

        return found, path, total_node_expansion

    MAX_G = 45

    @staticmethod
    def search(path, g, bound, model):

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
        successors = node.successors(model)

        for succ in successors:
            if succ not in path:
                path.append(succ)
                found = AStarSearch.search(path, g + 1, bound, model)
                if found:
                    return True

                path.pop()

        return found