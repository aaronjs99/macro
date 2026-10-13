import unittest
from unittest.mock import patch

from macro.graph import SystemGraph
from macro.ifmea import IFMEAEngine


class ExchangeStateTests(unittest.TestCase):
    def test_graph_rotation_removes_edges_from_the_previous_arrangement(self):
        graph = SystemGraph([list(range(6)), list(range(6, 18))], list(range(18)))
        self.assertIn(7, graph.get_neighbors(0))
        graph.rotate_layer(1, 1)
        self.assertNotIn(7, graph.get_neighbors(0))
        self.assertIn(17, graph.get_neighbors(0))

    def test_impossible_type_counts_fail_before_planning(self):
        with self.assertRaisesRegex(ValueError, "type|count"):
            IFMEAEngine([[2] * 6, [2] * 12])

    def test_reconfiguring_an_exchange_engine_does_not_replay_old_commands(self):
        engine = IFMEAEngine([[2, 1, 1, 1, 1, 1], [1] + [2] * 11])
        self.assertTrue(engine.run())
        engine.set_layer_config([[1] * 6, [2] * 12])
        self.assertEqual(engine.run(), [])

    def test_saturated_inner_layers_route_types_out_before_filling_the_next_layer(self):
        engine = IFMEAEngine([[3] * 6, [3] * 12, [1] * 6 + [2] * 12])
        exchange = engine.graph.exchange_positions
        count = 0

        def bounded_exchange(*args):
            nonlocal count
            count += 1
            if count > 200:
                self.fail(
                    "Planner repeatedly exchanges without reaching the destination layers"
                )
            exchange(*args)

        with patch.object(
            engine.graph, "exchange_positions", side_effect=bounded_exchange
        ):
            engine.run()
        self.assertEqual(engine.get_layer_config(), [[1] * 6, [2] * 12, [3] * 18])


if __name__ == "__main__":
    unittest.main()
