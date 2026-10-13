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
from abc import ABC, abstractmethod
from scipy.optimize import linear_sum_assignment

from macro.graph import SystemGraph
from macro.utils import NullLogger, check_intersection


class Auction(ABC):
    """
    Abstract base class for auction-based task allocation algorithms.

    Attributes:
        initial_positions (np.ndarray): Initial positions of agents.
        target_positions (np.ndarray): Target positions to assign.
        graph (SystemGraph): Agent communication graph.
        num_agents (int): Number of agents.
        max_iterations (int): Positive iteration limit; defaults to max(100, 100*n*n).

    Position arrays are copied so assignment never changes the caller's inputs.
    """

    def __init__(
        self,
        initial_positions: np.ndarray,
        target_positions: np.ndarray,
        graph: SystemGraph,
        *args,
        **kwargs,
    ):
        self.initial_positions = np.array(initial_positions, copy=True)
        self.target_positions = np.array(target_positions, copy=True)
        if (
            self.initial_positions.ndim != 2
            or self.target_positions.shape != self.initial_positions.shape
        ):
            raise ValueError(
                "Initial and target positions must have the same (agents, coordinates) shape"
            )
        if not np.all(np.isfinite(self.initial_positions)) or not np.all(
            np.isfinite(self.target_positions)
        ):
            raise ValueError(
                "Initial and target positions must contain finite coordinates"
            )
        self.graph = graph
        self.num_agents = self.initial_positions.shape[0]
        self.logger = kwargs.get("logger", NullLogger())
        self.max_iterations = kwargs.get(
            "max_iterations", max(100, 100 * self.num_agents**2)
        )
        if (
            isinstance(self.max_iterations, (bool, np.bool_))
            or not isinstance(self.max_iterations, (int, np.integer))
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")

    def _share_bids(self, prices, winners):
        """One synchronous neighbor exchange; relay the actual task winner ID."""
        next_prices, next_winners = prices.copy(), winners.copy()
        for agent in range(self.num_agents):
            neighbors = [agent] + self.graph.get_neighbors(agent)
            highest = np.max(prices[neighbors], axis=0)
            next_prices[agent] = highest
            next_winners[agent] = np.max(
                np.where(prices[neighbors] == highest, winners[neighbors], -1), axis=0
            )
        return next_prices, next_winners

    def _assignment_agreed(self, assignment, prices, winners):
        """Complete task claims and all local winner/price tables must agree."""
        return (
            np.array_equal(np.sort(assignment), np.arange(self.num_agents))
            and np.all(winners == winners[0])
            and np.all(prices == prices[0])
            and np.all(
                winners[np.arange(self.num_agents), assignment]
                == np.arange(self.num_agents)
            )
        )

    @abstractmethod
    def assign(self):
        """
        Abstract method to compute a task assignment.
        Must be implemented by subclasses.

        Returns:
            np.ndarray: New target positions after assignment.
        """
        return self.target_positions


class StandardAuction(Auction):
    """
    Implements the standard distributed auction using local Hungarian assignments.
    """

    def assign(self):
        """
        Assigns tasks using the Hungarian algorithm for each agent's neighborhood.

        Returns:
            np.ndarray: Updated target assignments.

        """
        for ego in range(self.num_agents):
            neighbors = self.graph.get_neighbors(ego)
            if not neighbors:
                continue
            neighbors_initial_positions = self.initial_positions[neighbors]
            neighbors_target_positions = self.target_positions[neighbors]
            cost_matrix = np.linalg.norm(
                neighbors_initial_positions[:, None, :] - neighbors_target_positions,
                axis=2,
            )
            if cost_matrix.size == 0:
                self.logger.debug(f"No neighbors for agent {ego}, skipping assignment")
                continue
            self.logger.debug(
                f"Agent {ego} neighbors: {neighbors}, cost matrix:\n{cost_matrix}"
            )
            row, col = linear_sum_assignment(cost_matrix)
            self.target_positions[np.asarray(neighbors)[row]] = (
                neighbors_target_positions[col]
            )
        return self.target_positions


class GreedyAuction(Auction):
    """
    Implements a greedy resolution of path conflicts by locally swapping assignments.
    """

    def assign(self):
        """
        Iteratively swaps assignments to resolve intersections.

        Returns:
            np.ndarray: Updated target assignments.

        Raises:
            RuntimeError: Intersections cause cycling or exceed the iteration limit.
        """
        assignment = np.arange(self.num_agents)
        seen = {tuple(assignment)}
        for _ in range(self.max_iterations):
            swapped = False
            for ego in range(self.num_agents):
                for other in self.graph.get_neighbors(ego):
                    if ego < other and check_intersection(
                        self.initial_positions[ego],
                        self.target_positions[ego],
                        self.initial_positions[other],
                        self.target_positions[other],
                    ):
                        self.target_positions[[ego, other]] = self.target_positions[
                            [other, ego]
                        ]
                        assignment[[ego, other]] = assignment[[other, ego]]
                        swapped = True
            if not swapped:
                self.logger.debug("No intersections found, assignment complete")
                return self.target_positions
            state = tuple(assignment)
            if state in seen:
                raise RuntimeError("Greedy auction did not converge: assignments cycle")
            seen.add(state)
        raise RuntimeError(
            f"Greedy auction did not converge within {self.max_iterations} iterations"
        )


class DistributedAuction(Auction):
    """
    Implements the distributed auction algorithm by Zavlanos et al.
    """

    def assign(self):
        """
        Runs distributed auction with local bidding and price updates.

        Returns:
            np.ndarray: Final assigned targets after convergence.

        Raises:
            RuntimeError: The graph is disconnected or the iteration limit is exceeded.
        """
        if self.num_agents < 2:
            return self.target_positions
        if not self.graph.is_connected():
            raise RuntimeError(
                "Distributed auction requires a connected communication graph"
            )

        # Maximize negative distance, using deterministic epsilon price increments.
        utility_matrix = -np.linalg.norm(
            self.initial_positions[:, None, :] - self.target_positions, axis=2
        )
        epsilon = max(float(np.ptp(utility_matrix)) / (100 * self.num_agents), 1e-12)
        assignment = np.full(self.num_agents, -1, dtype=int)
        prices = np.zeros((self.num_agents, self.num_agents))
        winners = np.full(prices.shape, -1, dtype=int)
        for _ in range(self.max_iterations):
            known_prices, known_winners = self._share_bids(prices, winners)
            prices, winners = known_prices.copy(), known_winners.copy()
            for agent in range(self.num_agents):
                task = assignment[agent]
                if task >= 0 and known_winners[agent, task] == agent:
                    continue
                net_utility = utility_matrix[agent] - known_prices[agent]
                task = int(np.argmax(net_utility))
                second_gain = np.max(np.delete(net_utility, task))
                increment = net_utility[task] - second_gain + epsilon
                prices[agent, task] = known_prices[agent, task] + increment
                winners[agent, task] = agent
                assignment[agent] = task
            if self._assignment_agreed(assignment, prices, winners):
                self.target_positions = self.target_positions[assignment]
                return self.target_positions
        raise RuntimeError(
            f"Distributed auction did not converge within {self.max_iterations} iterations"
        )


class ConsensusAuction(Auction):
    """
    Single-assignment Consensus-Based Auction Algorithm (CBAA) from Choi,
    Brunet & How (2009), alternating local bidding and winner consensus.
    """

    def assign(self):
        """
        Uses consensus mechanism to agree on unique task assignments.

        Returns:
            np.ndarray: Final target assignments after consensus.

        Raises:
            RuntimeError: The graph is disconnected or the iteration limit is exceeded.
        """
        if self.num_agents < 2:
            return self.target_positions
        if not self.graph.is_connected():
            raise RuntimeError(
                "Consensus auction requires a connected communication graph"
            )

        bid_utility = -np.linalg.norm(
            self.initial_positions[:, None, :] - self.target_positions, axis=2
        )
        # No-bid prices are -inf so zero and negative distance utilities are feasible.
        bid_price = np.full((self.num_agents, self.num_agents), -np.inf)
        bid_winner = np.full(bid_price.shape, -1, dtype=int)
        assignment = np.full(self.num_agents, -1, dtype=int)
        for _ in range(self.max_iterations):
            known_prices, known_winners = self._share_bids(bid_price, bid_winner)
            bid_price, bid_winner = known_prices.copy(), known_winners.copy()
            for agent in range(self.num_agents):
                task = assignment[agent]
                if task >= 0 and known_winners[agent, task] == agent:
                    continue
                assignment[agent] = -1
                utilities = bid_utility[agent]
                feasible = (utilities > known_prices[agent]) | (
                    (utilities == known_prices[agent]) & (agent > known_winners[agent])
                )
                if np.any(feasible):
                    task = int(np.argmax(np.where(feasible, utilities, -np.inf)))
                    assignment[agent] = task
                    bid_price[agent, task] = utilities[task]
                    bid_winner[agent, task] = agent
            if self._assignment_agreed(assignment, bid_price, bid_winner):
                self.target_positions = self.target_positions[assignment]
                return self.target_positions
        raise RuntimeError(
            f"Consensus auction did not converge within {self.max_iterations} iterations"
        )


class HybridAuction(Auction):
    """
    Combines greedy and distributed auction based on connectivity.

    If the graph is not connected, falls back to greedy conflict resolution.
    """

    def assign(self):
        """
        Choose between greedy and distributed auction depending on connectivity.

        Returns:
            np.ndarray: Final target assignments.
        """
        if self.num_agents < 2:
            return self.target_positions
        if not self.graph.is_connected():
            self.logger.debug("Graph is not connected, using greedy auction")
            return GreedyAuction(
                self.initial_positions,
                self.target_positions,
                self.graph,
                logger=self.logger,
                max_iterations=self.max_iterations,
            ).assign()
        else:
            self.logger.debug("Graph is connected, using distributed auction")
            return DistributedAuction(
                self.initial_positions,
                self.target_positions,
                self.graph,
                logger=self.logger,
                max_iterations=self.max_iterations,
            ).assign()


auction_map = {
    "Standard": StandardAuction,
    "Distributed": DistributedAuction,
    "Consensus": ConsensusAuction,
    "Greedy": GreedyAuction,
    "Hybrid": HybridAuction,
}
