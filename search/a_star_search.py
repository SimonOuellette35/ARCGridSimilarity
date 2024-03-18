import numpy as np
import time

VERBOSE = False
total_node_expansion = 0
iteration_node_expansion = 0
MAX_NODE_EXPANSIONS = 250000000
TIMEOUT = 864
HEURISTIC_TYPE = 'similarity'   # alternative: 'distance'

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
            tmp_children = []

            for idx in range(len(self.all_primitives)):

                tmp_grid = np.copy(self.current_grid)
                prim = self.all_primitives[idx]
                prim_func = prim[0]
                transformed_grid = self.apply_primitive(prim_func, tmp_grid)

                total_node_expansion += 1
                iteration_node_expansion += 1

                # a = np.reshape(tmp_grid, [-1])
                # b = np.reshape(transformed_grid, [-1])
                # if np.all(a == b):
                #     # don't bother with NOOPs
                #     continue

                child_node = Node(self.support_x, self.support_y,   # TODO: copies of the support set in each node is
                                                                    #  redundant
                                  transformed_grid, prim,
                                  self.all_primitives,              # TODO: also redundant?
                                  name="%s/%s" % (self.name, prim[1]))

                h_value = child_node.calc_h(model)

                print("\tchild: %s has h value %.2f" % (prim[1], h_value))

                if h_value == 0:
                    self.children.append(child_node)
                    return self.children

                if (time.time() - global_start_time) > TIMEOUT:
                    print("==> TIMEOUT!")
                    return self.children

                tmp_children.append([h_value, child_node])

            # sort children based on their h value
            children = sorted(tmp_children, key=lambda x: int(x[0]))

            order = ""
            child_nodes = []
            for c in children:
                child_nodes.append(c[1])
                order += "%s/" % c[1].parent_primitive[1]

            print("==> h ordering: ", order)
            self.children = child_nodes

        return self.children

    def calc_h(self, model):
        if self.h is not None:
            return self.h

        a = np.reshape(self.current_grid, [-1])
        b = np.reshape(self.support_y, [-1])
        found = np.all(a == b)

        if found:
            self.h = 0
            return self.h
        else:
            if model is None:
                return 1

            if self.h is None:
                # h must be the distance to the goal, while the model returns the similarity. Hence the 1 - similarity to
                # get the value of h.
                pred = model.get_similarity(np.array(self.current_grid), self.support_y)

                if HEURISTIC_TYPE == 'distance':
                    self.h = pred
                else:
                    self.h = int((1. - pred) * 10) + 1

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

global_start_time = 0

class AStarSearch():

    def __init__(self, max_depth):
        self.max_depth = max_depth

    def plan(self, root, model):
        global total_node_expansion
        global iteration_node_expansion
        global global_start_time

        total_node_expansion = 0
        iteration_node_expansion = 0

        global_start_time = time.time()

        bound = self.max_depth
        path = [root]

        found = False
        while not found and \
                total_node_expansion < MAX_NODE_EXPANSIONS and \
                time.time() - global_start_time <= TIMEOUT:

            found = self.search(path, 0, bound, model)
            bound += 1
            iteration_node_expansion = 0

        return found, path, total_node_expansion

    def search(self, path, g, bound, model):

        node = path[-1]
        tmp_h = 0
        if len(path) > 1:
            tmp_h = node.calc_h(model)

        print("==> Search(%s)" % path)
        f_cost = g + tmp_h
        print("f = ", f_cost)

        if len(path) > 1:
            if node.calc_h(model) == 0:
                print("\t===> FOUND GOAL after expanding %i nodes" % total_node_expansion)
                return True

        if g >= self.max_depth:
            if VERBOSE:
                print("\t==> Reached maximum depth %i. Aborting this path." % self.max_depth)
            return False

        if f_cost > bound:
            if VERBOSE:
                print("\t==> Cost exceeded bound, returning...")
            return False

        if total_node_expansion >= MAX_NODE_EXPANSIONS:
            print("\t===> Reached total_node_expansion of %i, returning..." % MAX_NODE_EXPANSIONS)
            return False

        found = False
        successors = node.successors(model)

        for succ in successors:
            if succ not in path:
                path.append(succ)
                found = self.search(path, g + 1, bound, model)
                if found:
                    return True

                if time.time() - global_start_time > TIMEOUT:
                    return found

                path.pop()

        return found