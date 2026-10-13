# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * Indian Institute of Technology, Bombay                  *
# *                                                         *
# * Author(s): Aaron John Sabu, Dwaipayan Mukherjee         *
# * Contact  : aaronjs@g.ucla.edu, dm@ee.iitb.ac.in         *
# *                                                         *
# ***********************************************************/


import networkx as nx

from macro.utils import to_layer_index, to_flat_index


class SystemGraph:
    def __init__(self, layer_config, flat_config):
        self.layer_config = layer_config
        self.flat_config = flat_config
        self.layer_lengths = [len(layer) for layer in layer_config]
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        self.graph.clear()
        num_agents = len(self.flat_config)
        self.graph.add_nodes_from(self.flat_config)

        for flat_idx in range(num_agents):
            layer_idx, pos_idx = to_layer_index(self.layer_lengths, flat_idx)
            agent_id = self.flat_config[flat_idx]

            # Sibling edges (wraparound)
            layer_len = self.layer_lengths[layer_idx]
            left_idx = (pos_idx - 1) % layer_len
            right_idx = (pos_idx + 1) % layer_len
            for sib_pos in [left_idx, right_idx]:
                sib_id = self.flat_config[
                    to_flat_index(self.layer_lengths, layer_idx, sib_pos)
                ]
                self.graph.add_edge(agent_id, sib_id)

            # Children (layer below)
            if layer_idx + 1 < len(self.layer_lengths):
                for offset in [-1, 0, 1]:
                    child_pos = (pos_idx * 2) + offset
                    if 0 <= child_pos < self.layer_lengths[layer_idx + 1]:
                        child_id = self.flat_config[
                            to_flat_index(self.layer_lengths, layer_idx + 1, child_pos)
                        ]
                        self.graph.add_edge(agent_id, child_id)

            # Parents (layer above)
            if layer_idx > 0:
                parent_positions = (
                    [pos_idx // 2]
                    if pos_idx % 2 == 0
                    else [(pos_idx - 1) // 2, (pos_idx + 1) // 2]
                )
                for parent_pos in parent_positions:
                    if parent_pos < self.layer_lengths[layer_idx - 1]:
                        parent_id = self.flat_config[
                            to_flat_index(self.layer_lengths, layer_idx - 1, parent_pos)
                        ]
                        self.graph.add_edge(agent_id, parent_id)

    def _update_flat_config(self):
        self.flat_config = [agent for layer in self.layer_config for agent in layer]
        self.layer_lengths = [len(layer) for layer in self.layer_config]

    def get_neighbors(self, agent_id):
        return list(self.graph.neighbors(agent_id))

    def get_all_neighbors(self):
        """
        Get a dictionary of all neighbors for each agent in the graph.
        Returns:
            dict: A dictionary where keys are agent IDs and values are lists of neighbor IDs.
        """
        return {
            agent_id: list(self.graph.neighbors(agent_id)) if agent_id in self.graph else []
            for agent_id in self.flat_config
        }

    def get_layer_lengths(self):
        return self.layer_lengths

    def get_layer_config(self):
        return self.layer_config

    def get_flat_config(self):
        return self.flat_config

    def get_adjacency_matrix(self):
        return nx.to_numpy_array(
            self.graph, nodelist=sorted(self.graph.nodes()), dtype=int
        )

    def is_connected(self):
        return nx.is_connected(self.graph)

    def as_networkx(self):
        return self.graph

    def get_config_distance(self, agent_id1, agent_id2):
        if self.graph.has_edge(agent_id1, agent_id2):
            return 1
        else:
            try:
                return nx.shortest_path_length(
                    self.graph, source=agent_id1, target=agent_id2
                )
            except nx.NetworkXNoPath:
                return float("inf")

    def rotate_layer(self, layer_idx, shift=1):
        """
        Rotate a specific layer in the graph.

        Parameters
        ----------
        layer_idx : int
            Index of the layer to rotate.
        shift: int, optional
            Number of positions to shift. Default is 1.
        """
        if layer_idx < 0 or layer_idx >= len(self.layer_config):
            raise ValueError("Layer index out of bounds.")

        if shift == 0:
            return
        direction = int(shift / abs(shift))

        for _ in range(abs(shift)):
            self.layer_config[layer_idx] = (
                self.layer_config[layer_idx][-direction:]
                + self.layer_config[layer_idx][:-direction]
            )

        self._update_flat_config()
        self._build_graph()

    def exchange_positions(self, pos1, pos2, pos3=None):
        """
        Exchange positions of two or three agents in the graph using (layer_idx, pos_idx).

        Parameters
        ----------
        pos1 : tuple[int, int]
            (layer_idx, pos_idx) of the first agent.
        pos2 : tuple[int, int]
            (layer_idx, pos_idx) of the second agent.
        pos3 : tuple[int, int], optional
            (layer_idx, pos_idx) of the third agent. If provided, positions of all three agents will be rotated.
        """
        l1, p1 = pos1
        l2, p2 = pos2

        if pos3 is not None:
            l3, p3 = pos3
            a1 = self.layer_config[l1][p1]
            a2 = self.layer_config[l2][p2]
            a3 = self.layer_config[l3][p3]

            self.layer_config[l1][p1] = a3
            self.layer_config[l2][p2] = a1
            self.layer_config[l3][p3] = a2
        else:
            a1 = self.layer_config[l1][p1]
            a2 = self.layer_config[l2][p2]

            self.layer_config[l1][p1] = a2
            self.layer_config[l2][p2] = a1

        self._update_flat_config()
        self._build_graph()
