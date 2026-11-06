"""Accounting, timing, intervention, adapter, and real-data boundary regressions.

Run: python -m unittest experiments.panic_reversal.test_framework -v
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .config import Config, MECHANISMS, CONTROLS
from .engine import Simulation, Result, clear_market, normalize_network, social_update
from .evaluation import conditional_targets, cost_adjust, paired_contrasts, summarize
from .observables import daily_features, tick_features
from .run import run_matrix
from .twinmarket_adapter import from_twinmarket, orders_to_twinmarket, validate_snapshot


def small(**kwargs):
    return replace(Config(agents=8, assets=4, steps=52, shock_step=20, horizons=(1, 5), stress_duration=6, recovery_steps=6), **kwargs)


def snapshot():
    profiles = [{"user_id": i, "current_cash": 1000.0,
                 "cur_positions": json.dumps({s: {"shares": 25, "ratio": 25} for s in ("A", "B", "C", "D")})} for i in range(4)]
    traits = {i: {"debt": 0.0, "risk": 1.0, "loss_aversion": 1.5, "social_weight": 0.2,
                  "info_weight": 0.1, "info_probability": 0.3, "trend": 0.2, "max_leverage": 2.0,
                  "information_source": "forum"} for i in range(4)}
    prices = {s: 100.0 for s in ("A", "B", "C", "D")}
    return from_twinmarket(profiles, [{"source": 0, "target": 1, "weight": 1.0}],
                           {i: "a qualitative belief" for i in range(4)},
                           lambda raw, p: {s: 0.0 for s in p}, traits, prices, "2023-01-03T15:00:00+08:00")


class EngineTests(unittest.TestCase):
    def test_reproducible_and_no_remote_services(self):
        with patch("socket.socket", side_effect=AssertionError("network forbidden")):
            a, b = Simulation(small()).run(), Simulation(small()).run()
        np.testing.assert_array_equal(a.prices, b.prices)
        self.assertEqual(a.market, b.market)

    def test_all_mechanisms_and_controls_conserve_resources(self):
        for mechanism in MECHANISMS:
            for control in CONTROLS:
                with self.subTest(mechanism=mechanism, control=control):
                    result = Simulation(small(), mechanism, control).run()
                    for key in ("max_cash_conservation_error", "max_stock_conservation_error", "max_debt_conservation_error"):
                        self.assertLess(result.audit[key], 1e-6)
                    self.assertGreaterEqual(result.audit["min_agent_cash"], -1e-8)
                    self.assertGreaterEqual(result.audit["min_dealer_cash"], -1e-8)
                    self.assertGreaterEqual(result.audit["min_inventory"], -1e-8)

    def test_no_shock_is_identical_before_event(self):
        for mechanism in MECHANISMS:
            a = Simulation(small(), mechanism, "baseline").run()
            b = Simulation(small(), mechanism, "no_shock").run()
            np.testing.assert_array_equal(a.prices[:21], b.prices[:21])

    def test_zero_severity_disables_every_intervention(self):
        for mechanism in MECHANISMS:
            a = Simulation(small(shock_size=0), mechanism).run()
            b = Simulation(small(shock_size=0), mechanism, "no_shock").run()
            np.testing.assert_array_equal(a.prices, b.prices)

    def test_future_simulation_length_does_not_change_prefix(self):
        # Fixed tapes must be stable under extension, including independent streams.
        a = Simulation(small()).run()
        b = Simulation(small(steps=60)).run()
        np.testing.assert_array_equal(a.prices, b.prices[:len(a.prices)])

    def test_social_is_lagged_and_isolated_nodes_receive_no_delta(self):
        graph = normalize_network([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        b, fear = np.array([[0.0], [1.0], [5.0]]), np.array([0.1, 0.9, 0.5])
        delta, peers = social_update(b, fear, graph, np.array([0.2, 0.2, 0.2]))
        np.testing.assert_allclose(delta[:, 0], [0.2, -0.2, 0])
        np.testing.assert_allclose(peers, [0.9, 0.1, 0])
        np.testing.assert_array_equal(b[:, 0], [0, 1, 5])

    def test_no_social_log_is_zero(self):
        logs = []
        Simulation(small(), "herding", "no_social").run(logs.append)
        self.assertTrue(all(np.count_nonzero(r["social_belief_delta"]) == 0 for r in logs))

    def test_information_loss_is_permanent(self):
        result = Simulation(small(), "information").run()
        values = [r["fundamental_latent"] for r in result.assets if r["asset"] == "S000"]
        self.assertAlmostEqual(values[19], 100)
        self.assertAlmostEqual(values[20], 100 * np.exp(-0.12))
        self.assertEqual(values[20], values[-1])

    def test_passive_selling_and_margin_repayments_are_distinct(self):
        liq = Simulation(small(), "liquidity").run()
        margin = Simulation(small(), "deleveraging").run()
        self.assertGreater(liq.market[-1]["withdrawals_cumulative"], 0)
        self.assertEqual(margin.market[-1]["withdrawals_cumulative"], 0)
        self.assertGreater(margin.market[-1]["debt_repaid_cumulative"], 0)
        self.assertGreater(sum(r["forced_sell_amount_latent"] for r in liq.market), 0)

    def test_dealer_rationing_keeps_unfilled_orders(self):
        orders = np.array([[-10.0], [0.0]])
        hold, cash = np.array([[10.0], [0.0]]), np.zeros(2)
        executed, forced, newcash, inventory, dealer_cash, fees, _, _ = clear_market(
            orders, -orders, hold, cash, np.array([0.0]), 100.0, np.array([100.0]), 0)
        np.testing.assert_allclose(executed[:, 0], [-1, 0])
        self.assertAlmostEqual(float(forced.sum()), 1)
        self.assertAlmostEqual(dealer_cash, 0)
        self.assertAlmostEqual(newcash.sum() + dealer_cash + fees.sum(), 100)
        self.assertAlmostEqual(float((hold + executed).sum() + inventory.sum()), 10)

    def test_actual_fills_prioritize_forced_obligation_within_account(self):
        executed, forced, *_ = clear_market(np.array([[-10.0]]), np.array([[2.0]]),
                                            np.array([[10.0]]), np.array([0.0]), np.array([0.0]),
                                            100.0, np.array([100.0]), 0)
        self.assertAlmostEqual(float(-executed[0, 0]), 1)
        self.assertAlmostEqual(float(forced[0, 0]), 1)

    def test_invalid_config_rejected(self):
        for kwargs in ({"horizons": (0,)}, {"shock_step": 5}, {"cost_bps_per_side": (-1,)},
                       {"shock_size": float("nan")}, {"recovery_steps": 0}, {"agent_fee_bps": 10000}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                small(**kwargs)


class EvaluationTests(unittest.TestCase):
    def test_signal_is_frozen_and_entry_is_delayed(self):
        c = small(lookback=1, horizons=(1,))
        prices = np.ones((c.steps + 1, 4)) * 100
        prices[1] = [90, 95, 105, 110]  # signal t=0: loser A, winner D
        prices[2] = [80, 90, 110, 120]  # t+1 entry; do NOT include this move
        prices[3] = [88, 90, 110, 108]  # forward +10% loser, -10% winner
        states = [{"step": 0, "latent_stage": "calm", "proxy_stage": "ordinary", "disagreement_bin_latent": "low", "belief_disagreement_latent": 0.0}]
        result = Result(c, "information", "baseline", prices, states, [], {"asset_codes": ["A", "B", "C", "D"]}, {})
        row = conditional_targets(result)[0]
        self.assertEqual((row["signal_step"], row["entry_step"], row["exit_step"]), (0, 1, 2))
        self.assertEqual((row["loser_assets"], row["winner_assets"]), ("A", "D"))
        self.assertAlmostEqual(row["spread_gross"], 0.2)
        prices[3] = [64, 90, 110, 144]
        changed = conditional_targets(result)[0]
        self.assertEqual(row["loser_assets"], changed["loser_assets"])

    def test_censoring_and_cost_legs(self):
        result = Simulation(small()).run()
        rows = conditional_targets(result)
        self.assertTrue(all(r["exit_step"] < result.config.steps for r in rows))
        adjusted = cost_adjust(rows[0], 10)
        self.assertAlmostEqual(adjusted["spread_gross"] - adjusted["spread_net_theoretical"], 0.004)
        self.assertAlmostEqual(adjusted["long_only_excess_gross"] - adjusted["long_only_excess_net_proxy"], 0.002)

    def test_overlap_does_not_inflate_seed_sample_size(self):
        row = conditional_targets(Simulation(small()).run())[0]
        data = [{**row, "seed": seed} for seed in (1, 2) for _ in range(20)]
        for group in summarize(data, (10,)):
            self.assertEqual(group["n_seeds"], 2)
            self.assertEqual(group["n_overlapping_cohorts"], 40)
            self.assertIsNone(group["seed_mc_ci_low"])

    def test_paired_differences_use_common_seed_and_fixed_time(self):
        row = next(r for r in conditional_targets(Simulation(small()).run()) if r["fixed_event_signal"])
        rows = [{**row, "seed": seed, "control": control, "spread_gross": 0.03 if control == "baseline" else 0.01}
                for seed in (42, 43, 44) for control in ("baseline", "no_social")]
        contrast = paired_contrasts(rows, (0, 30))
        self.assertEqual(len(contrast), 2)
        for r in contrast:
            self.assertEqual(r["n_seed_pairs"], 3)
            self.assertAlmostEqual(r["baseline_minus_control_spread"], 0.02)


class AdapterTests(unittest.TestCase):
    def test_actual_profile_schema_and_directed_influence(self):
        snap = snapshot()
        sim = Simulation(small(agents=4), snapshot=snap)
        self.assertEqual(sim.ids, ["0", "1", "2", "3"])
        self.assertEqual(sim.network[1, 0], 1)
        self.assertEqual(sim.network[0, 1], 0)
        result = sim.run()
        self.assertEqual(result.initial["holdings"][0], [25, 25, 25, 25])

    def test_text_and_missing_debt_cannot_be_implicitly_classified(self):
        snap = snapshot()
        del snap["profiles"][0]["debt"]
        with self.assertRaises(ValueError):
            validate_snapshot(snap)
        snap = snapshot()
        snap["profiles"][0]["belief_numeric"] = "panic"
        with self.assertRaises(ValueError):
            validate_snapshot(snap)

    def test_cash_only_investor_can_buy(self):
        snap = snapshot()
        snap["profiles"][0]["cur_positions"] = {}
        sim = Simulation(small(agents=4), snapshot=snap)
        self.assertTrue(np.all(sim.preference[0] == 0.25))

    def test_snapshot_preferences_use_value_not_share_counts(self):
        snap = snapshot()
        snap["prices"]["A"] = 10.0
        sim = Simulation(small(agents=4), snapshot=snap)
        np.testing.assert_allclose(sim.preference[0], np.array([10, 100, 100, 100]) / 310)

    def test_export_schema_and_rounding(self):
        exported = orders_to_twinmarket(["x"], ["A", "B", "C"], [[250, -199, 99]], [10, 20, 30])
        self.assertEqual([r["amount"] for r in exported], [200, 100])
        self.assertEqual([r["direction"] for r in exported], ["buy", "sell"])
        self.assertEqual(set(exported[0]), {"user_id", "stock_code", "direction", "amount", "target_price"})

    def test_bad_graph_and_duplicate_ids_rejected(self):
        snap = snapshot()
        snap["profiles"][1]["user_id"] = "0"
        with self.assertRaises(ValueError):
            validate_snapshot(snap)
        with self.assertRaises(ValueError):
            normalize_network([[0, -1], [1, 0]])


class ObservableTests(unittest.TestCase):
    def test_daily_prefix_and_no_filled_missing_returns(self):
        rows = [{"date": day, "asset": code, "close": price, "adj_close": price, "volume": 100, "amount": 100 * price}
                for day, code, price in (("2023-01-03", "A", 100), ("2023-01-03", "B", 100),
                                         ("2023-01-04", "A", 90), ("2023-01-05", "A", 99), ("2023-01-05", "B", 80))]
        result = daily_features(rows)
        self.assertEqual(result[:2], daily_features(rows[:3]))
        self.assertEqual(result[-1]["n_return_assets"], 1)
        self.assertEqual(result[-1]["coverage"], 0.5)
        self.assertAlmostEqual(result[-1]["market_return_equal_weight"], 0.1)

    def test_tick_unknown_side_stays_missing_and_timestamps_are_explicit(self):
        row = {"timestamp": "2023-01-03T09:31:00+08:00", "asset": "A", "price": 100, "volume": 10, "aggressor": ""}
        result = tick_features([row])[0]
        self.assertIsNone(result["aggressive_sell_share"])
        self.assertEqual(result["aggressor_coverage"], 0)
        with self.assertRaises(ValueError):
            tick_features([{**row, "timestamp": "2023-01-03T09:31:00"}])

    def test_tick_concentration_and_no_future_daily_total(self):
        rows = [{"timestamp": f"2023-01-03T09:{minute}:00+08:00", "asset": code, "price": price, "volume": 10, "aggressor": "-1"}
                for minute, code, price in (("31", "A", 100), ("32", "B", 100), ("36", "A", 90))]
        prefix = tick_features(rows[:2])
        all_ = tick_features(rows)
        self.assertEqual(prefix, all_[:1])
        self.assertAlmostEqual(all_[0]["sell_concentration_assets"], 0.5)
        self.assertAlmostEqual(all_[0]["sell_time_hhi_day_to_date"], 1)

    def test_invalid_or_duplicate_daily_rows_fail(self):
        row = {"date": "2023-01-03", "asset": "A", "close": 100, "adj_close": 100, "volume": 0, "amount": 0}
        with self.assertRaises(ValueError):
            daily_features([row, row])
        with self.assertRaises(ValueError):
            daily_features([{**row, "close": -1}])


class RunnerTests(unittest.TestCase):
    def test_cli_artifacts_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new"
            summary = run_matrix(small(), [42], output, ("herding",), ("baseline", "no_social"), logs="example")
            self.assertEqual(summary["runs"], 2)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["completed"])
            self.assertTrue((output / "paired_effects.csv").stat().st_size > 0)
            self.assertTrue((output / "agents_herding_baseline_42.jsonl").is_file())
            with self.assertRaises(FileExistsError):
                run_matrix(small(), [42], output, ("herding",), ("baseline",), logs="none")


if __name__ == "__main__":
    unittest.main()
