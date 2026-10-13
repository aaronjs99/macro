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

import numpy as np
from collections import Counter
from numbers import Integral

from macro.graph import SystemGraph
from macro.utils import NullLogger


class IFMEAEngine:
    """
    Initialize the IFMEAEngine with a given layer configuration and optional layer types.

    Parameters
    ----------
    layer_config : list[list[int]]
        The nested list representing agents in concentric layers.
    layer_type : list[int], optional
        List of target types corresponding to each layer.
    *args, **kwargs :
        Optional keyword arguments including:
        - graph: a pre-initialized SystemGraph object.
    """

    def __init__(self, layer_config=None, layer_type=None, *args, **kwargs):
        """
        Initialize the IFMEAEngine with layer configuration and type.
        """
        self.layer_config = layer_config
        self.layer_type = layer_type
        self.commands = []
        self.logger = kwargs.get("logger", NullLogger())

        if layer_config is not None:
            self.set_layer_config(layer_config, layer_type, **kwargs)

    def set_layer_config(self, layer_config, layer_type=None, **kwargs):
        """
        Set the layer configuration for the IFMEAEngine.

        Parameters
        ----------
        layer_config : list[list[int]]
            The new layer configuration to set.
        """
        self.layer_config = layer_config
        self.layer_type = layer_type or list(range(len(layer_config)))
        self._validate_inputs()
        self.layer_config = [layer.copy() for layer in layer_config]
        self.commands = []

        self.flat_config = [agent for layer in self.layer_config for agent in layer]
        self.graph = kwargs.get(
            "graph", SystemGraph(self.layer_config, self.flat_config)
        )
        self.layer_lengths = self.graph.get_layer_lengths()

    def _validate_inputs(self):
        """Validate input types and lengths."""
        if not self.layer_config:
            raise ValueError("Layer configuration cannot be empty.")
        if not isinstance(self.layer_config, list) or not all(
            isinstance(layer, list) for layer in self.layer_config
        ):
            raise TypeError("Layer configuration must be a list of lists.")
        if not isinstance(self.layer_type, list):
            raise TypeError("Layer type must be a list of integers.")
        if len(self.layer_config) != len(self.layer_type):
            raise ValueError(
                "Layer configuration and layer type must have the same number of layers."
            )
        counts = Counter(
            agent_type for layer in self.layer_config for agent_type in layer
        )
        if any(
            isinstance(agent_type, bool)
            or not isinstance(agent_type, Integral)
            or not 1 <= agent_type <= len(self.layer_config)
            for agent_type in counts
        ):
            raise ValueError(
                "Agent types must be integers from 1 to the number of layers"
            )
        if any(
            counts[i + 1] != len(layer) for i, layer in enumerate(self.layer_config)
        ):
            raise ValueError(
                "Each agent type count must equal the size of its destination layer"
            )

    def _find_nearest_mismatch(self, outer_layer_idx, inner_layer_idx, curr_idx):
        """
        Find the closest agent in the next layer (layer_idx+1) that has the lowest type,
        starting from a target index calculated from the current position.

        Parameters
        ----------
        layer_idx : int
            Index of the current inner layer.
        pos_idx : int
            Index of the current agent within the inner layer.

        Returns
        -------
        tuple[int, int]
            The index of the mismatch in the next layer, and the target index to align with.
        """
        curr_layer = self.graph.get_layer_config()[inner_layer_idx]
        next_layer = self.graph.get_layer_config()[inner_layer_idx + 1]
        len_curr = len(curr_layer)
        len_next = len(next_layer)

        next_idx = int(curr_idx * len_next / len_curr)
        mismatch_idx = None

        for dist in range(int(len_next / 2) + 1):
            upper = (next_idx + dist) % len_next
            lower = (next_idx - dist) % len_next
            if next_layer[upper] < outer_layer_idx + 1:
                mismatch_idx = upper
                break
            elif next_layer[lower] < outer_layer_idx + 1:
                mismatch_idx = lower
                break

        return mismatch_idx, next_idx

    def step_layer(self, outer_layer_idx):
        """
        Process a single outer layer, identifying and resolving mismatches between it
        and the layer directly beneath by issuing rotations or exchanges.

        Parameters
        ----------
        outer_layer_idx : int
            The index of the outer layer to process.
        """
        inner_layer_idx = 0
        seen = set()

        while inner_layer_idx != outer_layer_idx:
            state = (
                inner_layer_idx,
                tuple(tuple(layer) for layer in self.graph.get_layer_config()),
            )
            if state in seen:
                raise RuntimeError(
                    "IFMEA planner repeated a configuration without progress"
                )
            seen.add(state)
            self.logger.debug(
                f"Processing inner layer {inner_layer_idx} for outer layer {outer_layer_idx}"
            )
            self.logger.debug(
                f"Configuration before processing: {self.graph.get_layer_config()}"
            )
            try:
                # Find position of agent needing to move up
                curr_idx = self.graph.get_layer_config()[inner_layer_idx].index(
                    outer_layer_idx + 1
                )
            except ValueError:
                inner_layer_idx += 1
                continue

            # Find closest mismatch position in the layer above
            mismatch_idx, next_idx = self._find_nearest_mismatch(
                outer_layer_idx, inner_layer_idx, curr_idx
            )
            if mismatch_idx is None:
                # A saturated next layer must first move its own outer-type agents onward.
                inner_layer_idx += 1
                continue

            # Rotate to align source and target
            shift = (next_idx - mismatch_idx) % self.layer_lengths[inner_layer_idx + 1]
            self.logger.debug(
                f"Shifting layer {inner_layer_idx + 1} by {shift} positions to align with target."
            )
            if shift != 0:
                self.graph.rotate_layer(inner_layer_idx + 1, shift)
                self.commands.append(("R", inner_layer_idx + 1, shift))

            # Perform exchange of positions
            third_dir = (
                next_idx / self.layer_lengths[inner_layer_idx + 1]
                < curr_idx / self.layer_lengths[inner_layer_idx]
            )
            if third_dir:
                # Mismatch is to the left of target
                third_idx = (curr_idx - 1) % self.layer_lengths[inner_layer_idx]
            else:
                # Mismatch is to the right of target
                third_idx = (curr_idx + 1) % self.layer_lengths[inner_layer_idx]

            self.logger.debug(
                f"Exchanging positions: {curr_idx} in layer {inner_layer_idx} of type {self.layer_config[inner_layer_idx][curr_idx]}, {next_idx} in layer {inner_layer_idx + 1} of type {self.layer_config[inner_layer_idx + 1][next_idx]}, and {third_idx} in layer {inner_layer_idx} of type {self.layer_config[inner_layer_idx][third_idx]}"
            )
            self.graph.exchange_positions(
                (inner_layer_idx, curr_idx),
                (inner_layer_idx + 1, next_idx),
                (inner_layer_idx, third_idx),
            )
            self.commands.append(
                (
                    "E",
                    (inner_layer_idx, curr_idx),
                    (inner_layer_idx + 1, next_idx),
                    (inner_layer_idx, third_idx),
                )
            )
            self.logger.debug(
                f"Configuration after processing: {self.graph.get_layer_config()}"
            )
            inner_layer_idx = 0

    def run(self):
        """
        Run the IFMEA algorithm from the outermost to the innermost layer.

        Returns
        -------
        list[tuple]
            A list of commands executed by the algorithm.
        """
        self.logger.info("Starting IFMEA algorithm run.")
        if not self.layer_config:
            self.logger.warning("Layer configuration is empty, no commands generated.")
            return self.commands
        self.logger.info(f"Initial layer configuration: {self.layer_config}")
        if not self.graph.is_connected():
            self.logger.error("Graph is not connected, cannot run IFMEA.")
            return self.commands
        for outer_layer_idx in reversed(range(1, len(self.layer_config))):
            self.logger.debug(f"Processing outer layer {outer_layer_idx}")
            self.step_layer(outer_layer_idx)
        self.logger.info(f"Final layer configuration: {self.graph.get_layer_config()}")
        if not self.commands:
            self.logger.warning("No commands generated during IFMEA run.")
        else:
            self.logger.info(f"Total commands generated: {len(self.commands)}")
        self.logger.info(f"Commands: {self.commands}")
        return self.commands

    def get_commands(self):
        """
        Get the list of commands generated by the IFMEA algorithm.

        Returns
        -------
        list[tuple]
            The list of commands.
        """
        return self.commands

    def get_layer_config(self):
        """
        Get the current layer configuration of the graph.

        Returns
        -------
        list[list[int]]
            The current layer configuration.
        """
        return self.graph.get_layer_config()


if __name__ == "__main__":
    import logging

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    counts = {
        1: 6,
        2: 12,
        3: 18,
        4: 24,
        5: 30,
        6: 30,
        7: 24,
        8: 18,
        9: 12,
        10: 6,
        11: 1,
        12: 1,
    }
    layer_sizes = [counts[i] for i in range(1, len(counts) + 1)]

    ifmea_engine = IFMEAEngine(logger=logger)
    all_agents = [
        agent_type for agent_type, count in counts.items() for _ in range(count)
    ]

    for i in range(1):
        np.random.shuffle(all_agents)

        init_layer_config = []
        idx = 0
        for size in layer_sizes:
            init_layer_config.append(all_agents[idx : idx + size])
            idx += size

        ifmea_engine.set_layer_config(init_layer_config)
        ifmea_engine.run()
