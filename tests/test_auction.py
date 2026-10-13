"""Auctions preserve target rows and resolve competing task claims honestly."""

import subprocess
import sys
import unittest

import networkx as nx
import numpy as np

from macro.auction import (
    ConsensusAuction,
    DistributedAuction,
    GreedyAuction,
    HybridAuction,
    StandardAuction,
    auction_map,
)
from macro.graph import SystemGraph


def system_graph(num_agents, edges=None):
    agents = list(range(num_agents))
    graph = SystemGraph([agents], agents)
    graph.graph = nx.complete_graph(num_agents)
    if edges is not None:
        graph.graph.remove_edges_from(list(graph.graph.edges))
        graph.graph.add_edges_from(edges)
    return graph


class AuctionAssignmentTests(unittest.TestCase):
    def assert_target_permutation(self, assigned, targets):
        self.assertEqual(assigned.shape, targets.shape)
        self.assertCountEqual(list(map(tuple, assigned)), list(map(tuple, targets)))

    def test_standard_prefers_shorter_travel(self):
        initial = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        targets = initial[::-1].copy()
        assigned = StandardAuction(initial, targets, system_graph(3)).assign()
        np.testing.assert_array_equal(assigned, initial)

    def test_auctions_preserve_caller_arrays(self):
        for name, strategy in auction_map.items():
            with self.subTest(strategy=name):
                initial = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
                targets = initial[::-1].copy()
                initial_before, targets_before = initial.copy(), targets.copy()
                strategy(initial, targets, system_graph(3)).assign()
                np.testing.assert_array_equal(initial, initial_before)
                np.testing.assert_array_equal(targets, targets_before)

    def test_greedy_swaps_whole_rows_for_agent_ids_above_dimension(self):
        initial = np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        )
        targets = np.array(
            [[2.0, 2.0, 0.0], [11.0, 0.0, 0.0], [21.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        )
        assigned = GreedyAuction(initial, targets, system_graph(4, [(0, 3)])).assign()
        np.testing.assert_array_equal(assigned, targets[[3, 1, 2, 0]])
        self.assert_target_permutation(assigned, targets)

    def test_distributed_resolves_duplicate_task_claims(self):
        initial = np.zeros((3, 3))
        targets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        assigned = DistributedAuction(initial, targets, system_graph(3)).assign()
        self.assert_target_permutation(assigned, targets)

    def test_bidding_auctions_prefer_nearby_tasks_including_zero_distance(self):
        initial = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        targets = initial[::-1].copy()
        for strategy in (DistributedAuction, ConsensusAuction, HybridAuction):
            with self.subTest(strategy=strategy.__name__):
                assigned = strategy(initial, targets, system_graph(3)).assign()
                np.testing.assert_array_equal(assigned, initial)

    def test_distributed_strategies_reject_disconnected_graphs(self):
        initial = np.zeros((3, 3))
        targets = np.eye(3)
        for strategy in (DistributedAuction, ConsensusAuction):
            with self.subTest(strategy=strategy.__name__):
                with self.assertRaisesRegex(RuntimeError, "connected"):
                    strategy(initial, targets, system_graph(3, [(0, 1)])).assign()

    def test_winners_propagate_across_a_sparse_path(self):
        initial = np.zeros((5, 3))
        targets = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        )
        graph = system_graph(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
        for strategy in (DistributedAuction, ConsensusAuction, HybridAuction):
            with self.subTest(strategy=strategy.__name__):
                assigned = strategy(initial, targets, graph).assign()
                self.assert_target_permutation(assigned, targets)

    def test_hybrid_resolves_conflicts_locally_when_disconnected(self):
        initial = np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        )
        targets = np.array(
            [[2.0, 2.0, 0.0], [11.0, 0.0, 0.0], [21.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        )
        assigned = HybridAuction(initial, targets, system_graph(4, [(0, 3)])).assign()
        np.testing.assert_array_equal(assigned, targets[[3, 1, 2, 0]])

    def test_auctions_preserve_multiplicity_of_identical_targets(self):
        initial = np.zeros((4, 3))
        targets = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        for name, strategy in auction_map.items():
            with self.subTest(strategy=name):
                assigned = strategy(initial, targets, system_graph(4)).assign()
                self.assert_target_permutation(assigned, targets)

    def test_empty_and_single_agent_inputs(self):
        for count in (0, 1):
            initial = np.zeros((count, 3))
            targets = np.ones((count, 3))
            for name, strategy in auction_map.items():
                with self.subTest(strategy=name, count=count):
                    assigned = strategy(initial, targets, system_graph(count)).assign()
                    self.assert_target_permutation(assigned, targets)

    def test_incompatible_position_arrays_are_rejected_before_assignment(self):
        initial = np.zeros((3, 3))
        for targets in (np.zeros((2, 3)), np.zeros((3, 2)), np.zeros(3)):
            with self.subTest(shape=targets.shape):
                with self.assertRaisesRegex(ValueError, "shape"):
                    StandardAuction(initial, targets, system_graph(3))
        for value in (np.nan, np.inf):
            targets = np.zeros((3, 3))
            targets[0, 0] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    DistributedAuction(initial, targets, system_graph(3))

    def test_invalid_iteration_budgets_are_rejected(self):
        for budget in (0, -1, 1.5, True):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, "max_iterations"):
                    GreedyAuction(
                        np.zeros((1, 3)),
                        np.ones((1, 3)),
                        system_graph(1),
                        max_iterations=budget,
                    )

    def test_distributed_iteration_budget_is_reported(self):
        initial = np.zeros((3, 3))
        targets = np.eye(3)
        for strategy in (DistributedAuction, ConsensusAuction):
            with self.subTest(strategy=strategy.__name__):
                with self.assertRaisesRegex(RuntimeError, "converge"):
                    strategy(
                        initial, targets, system_graph(3), max_iterations=1
                    ).assign()

    def test_greedy_unresolvable_intersections_terminate(self):
        # Both paths start at the same point, so swapping cannot remove that intersection.
        code = """
import numpy as np
from macro.auction import GreedyAuction
from macro.graph import SystemGraph
initial = np.zeros((2, 3))
targets = np.array([[1., 0., 0.], [0., 1., 0.]])
graph = SystemGraph([[0, 1]], [0, 1])
try:
    GreedyAuction(initial, targets, graph, max_iterations=2).assign()
except RuntimeError:
    pass
else:
    raise AssertionError('Expected an explicit nonconvergence error')
"""
        try:
            completed = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, timeout=5
            )
        except subprocess.TimeoutExpired:
            self.fail("Greedy did not respect a bounded convergence contract")
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
