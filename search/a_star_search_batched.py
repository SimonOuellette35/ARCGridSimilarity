import numpy as np
import time

VERBOSE = False
total_node_expansion = 0
iteration_node_expansion = 0
MAX_NODE_EXPANSIONS = 250000000
TIMEOUT = 30

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
        self.num_prims = len(self.all_primitives)

        self.batched_target_grids = []
        for _ in range(len(self.all_primitives)):
            for idx in range(support_y.shape[0]):
                self.batched_target_grids.append(support_y[idx])

        self.batched_target_grids = np.reshape(self.batched_target_grids, [len(self.batched_target_grids), -1])

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

            intermediate_grids = []
            for idx in range(len(self.all_primitives)):

                prim = self.all_primitives[idx]
                prim_func = prim[0]

                # NOTE: this assumes all primitives return a copy of the grid and do not touch the original
                transformed_grids = self.apply_primitive(prim_func, self.current_grid)

                total_node_expansion += 1
                iteration_node_expansion += 1

                child_node = Node(self.support_x, self.support_y,
                                  transformed_grids, prim,
                                  self.all_primitives,
                                  name="%s/%s" % (self.name, prim[1]))

                # first look for exact matches...
                a = np.reshape(transformed_grids, [-1])
                b = np.reshape(self.support_y, [-1])
                found = np.all(a == b)

                if found:
                    child_node.h = 0
                    self.children.append(child_node)
                    return self.children

                for tg in transformed_grids:
                    intermediate_grids.append(tg)

                tmp_children.append([np.inf, child_node])

            intermediate_grids = np.reshape(intermediate_grids, [len(intermediate_grids), -1])
            sim_preds = model.get_batched_similarity(intermediate_grids, self.batched_target_grids)

            k = self.support_x.shape[0]
            median_sims_per_prim = np.median(np.reshape(sim_preds, [self.num_prims, k]), axis=-1)

            h_per_prim = (1. - median_sims_per_prim) * 10

            for prim_idx, h_val in enumerate(h_per_prim):
                h = int(h_val) + 1
                tmp_children[prim_idx][1].h = h
                tmp_children[prim_idx][0] = h

            if (time.time() - global_start_time) > TIMEOUT:
                print("==> TIMEOUT!")
                return self.children

            # sort children based on their h value
            tmp_children.sort(key=lambda x: x[0])
            self.children = [child[1] for child in tmp_children]

        return self.children

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
            tmp_h = node.h

        f_cost = g + tmp_h

        if len(path) > 1:
            if node.h == 0:
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