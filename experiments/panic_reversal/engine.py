"""Finite-budget agents trading with each other and a finite-inventory dealer.

One step is one stylized trading day, not a Tick or a full exchange simulator.
Private value estimates and emotions are latent variables. All updates use only
the previous peer state. Price recovery is never prescribed as an outcome.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .config import Config, CONTROLS, MECHANISMS


@dataclass
class Result:
    config: Config
    mechanism: str
    control: str
    prices: np.ndarray
    market: list
    assets: list
    initial: dict
    audit: dict


def normalize_network(weights):
    weights = np.asarray(weights, dtype=float).copy()
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("Network must be square")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Network weights must be finite and nonnegative")
    np.fill_diagonal(weights, 0.0)
    totals = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)


def social_update(belief, fear, network, weights):
    """Synchronous, lagged neighbors; isolated nodes receive zero contribution."""
    active = network.sum(axis=1) > 0
    peer_belief = network @ belief
    peer_fear = network @ fear
    delta = weights[:, None] * (peer_belief - belief) * active[:, None]
    return delta, peer_fear * active


def clear_market(orders, forced_orders, holdings, cash, dealer_inventory, dealer_cash,
                 prices, fee_rate):
    """Cross agents first, then ration residual orders by dealer resources.

    Every executed share and unit of cash has a counterparty. A quote can move
    on unfilled pressure; this is a reduced-form dealer, not a limit order book.
    Forced sells receive priority within each agent, not across agents.
    """
    buy = np.maximum(orders, 0)
    buy_bill = buy @ prices * (1 + fee_rate)
    buy *= np.minimum(1, np.divide(cash, buy_bill, out=np.ones_like(cash), where=buy_bill > 0))[:, None]
    sell = np.minimum(np.maximum(-orders, 0), holdings)
    b, s = buy.sum(axis=0), sell.sum(axis=0)
    crossed = np.minimum(b, s)
    dealer_sell = np.minimum(np.maximum(b - s, 0), dealer_inventory)
    dealer_buy = np.maximum(s - b, 0)
    bill = float(dealer_buy @ prices)
    # Receipts from sales of inventory can finance purchases in this auction.
    budget = dealer_cash + float(dealer_sell @ prices)
    if bill > budget:
        dealer_buy *= max(0.0, budget) / bill
    bfill = np.divide(crossed + dealer_sell, b, out=np.zeros_like(b), where=b > 0)
    sfill = np.divide(crossed + dealer_buy, s, out=np.zeros_like(s), where=s > 0)
    bought, sold = buy * bfill, sell * sfill
    executed = bought - sold
    forced_filled = np.minimum(forced_orders, sold)
    fees = (bought + sold) @ prices * fee_rate
    new_cash = cash - executed @ prices - fees
    new_dealer_inventory = dealer_inventory - executed.sum(axis=0)
    new_dealer_cash = dealer_cash + float(executed.sum(axis=0) @ prices)
    return executed, forced_filled, new_cash, new_dealer_inventory, new_dealer_cash, fees, bought, sold


class Simulation:
    def __init__(self, config: Config, mechanism="information", control="baseline", snapshot=None):
        if mechanism not in MECHANISMS or control not in CONTROLS:
            raise ValueError("Unknown mechanism or control")
        self.c, self.mechanism, self.control = config, mechanism, control
        init_seed, noise_seed, arrival_seed, market_seed, emotion_seed = np.random.SeedSequence(config.seed).spawn(5)
        rng = np.random.default_rng(init_seed)
        n, k, t = config.agents, config.assets, config.steps
        self.codes = [f"S{j:03d}" for j in range(k)]
        self.ids = [f"agent_{i:03d}" for i in range(n)]
        self.price = np.full(k, 100.0)
        equity = rng.lognormal(np.log(50000), 0.5, n)
        self.cash = equity * rng.uniform(0.03, 0.15, n)
        self.debt = equity * rng.choice([0.0, 0.0, 0.75], n)
        weights = rng.dirichlet(np.full(k, 1.5), n)
        self.holdings = weights * (equity + self.debt - self.cash)[:, None] / self.price
        self.info_weight = rng.uniform(0.06, 0.22, n)
        self.social_weight = rng.uniform(0.08, 0.35, n)
        self.risk = rng.uniform(0.7, 2.0, n)
        self.loss_aversion = rng.uniform(1.0, 2.5, n)
        self.trend = rng.uniform(-0.5, 1.2, n)
        # Source categories determine information arrival, not knowledge of V.
        self.source = np.arange(n) % 3  # official / mixed / forum
        self.info_probability = np.array([0.7, 0.3, 0.1])[self.source]
        self.max_leverage = rng.uniform(2.0, 2.6, n)
        self.belief = np.log(self.price)[None, :] + rng.normal(0, 0.025, (n, k))
        net = np.zeros((n, n))
        for i in range(n):
            for offset in (-2, -1, 1, 2):
                net[i, (i + offset) % n] = 1
        net[rng.random((n, n)) < 0.035] = 0.5
        self.network = normalize_network(net)
        self.snapshot_metadata = None
        if snapshot is not None:
            self._load_snapshot(snapshot)
        if control == "homogeneous":
            for name in ("cash", "debt", "holdings", "info_weight", "social_weight", "risk", "loss_aversion", "trend", "max_leverage", "belief", "info_probability"):
                x = getattr(self, name)
                x[:] = np.mean(x, axis=0)
            self.source[:] = 1
        if control in ("no_social", "simple_rule"):
            self.social_weight[:] = 0
        self.fundamental = np.log(self.price).copy()
        self.fear = np.zeros(n)
        held_values = self.holdings * self.price
        total_held = held_values.sum(axis=1, keepdims=True)
        self.preference = np.divide(held_values, total_held, out=np.full_like(self.holdings, 1 / k), where=total_held > 0)
        # Common random numbers: all branches consume identical fixed tapes.
        self.observation_noise = np.random.default_rng(noise_seed).normal(0, 0.04, (t, n, k))
        self.observation_arrival = np.random.default_rng(arrival_seed).random((t, n))
        self.market_noise = np.random.default_rng(market_seed).normal(0, 0.0015, (t, k))
        self.emotion_tape = np.random.default_rng(emotion_seed).beta(1.5, 6.0, (t, n)) * 0.9
        self.dealer_inventory = np.maximum(self.holdings.sum(axis=0) * 0.4, 100.0)
        self.dealer_cash = float(self.holdings.sum(axis=0) @ self.price) * 0.7
        self.supply = self.holdings.sum(axis=0) + self.dealer_inventory
        self.initial_equity = self.cash + self.holdings @ self.price - self.debt
        self.initial_cash = float(self.cash.sum() + self.dealer_cash)
        self.initial_debt = float(self.debt.sum())
        self.withdrawn = self.lender_cash = self.fee_sink = 0.0
        self.pending = np.zeros(n)
        self.peak_wealth = self.initial_equity.copy()
        self.ever_panicked = np.zeros(n, dtype=bool)
        self.target_agents = np.arange(n) % 4 == 0
        self.target_assets = np.linspace(1.0, 0.25, k)

    def _load_snapshot(self, snapshot):
        # The adapter validates raw TwinMarket fields without importing its LLM.
        from .twinmarket_adapter import validate_snapshot
        validate_snapshot(snapshot)
        codes = list(snapshot["prices"])
        profiles = snapshot["profiles"]
        if len(profiles) != self.c.agents or len(codes) != self.c.assets:
            raise ValueError("Snapshot dimensions must match Config agents/assets")
        self.codes, self.ids = codes, [str(p["user_id"]) for p in profiles]
        self.price = np.array([snapshot["prices"][s] for s in codes], dtype=float)
        self.cash = np.array([p["current_cash"] for p in profiles], dtype=float)
        self.debt = np.array([p["debt"] for p in profiles], dtype=float)
        self.holdings = np.array([[p["cur_positions"].get(s, {}).get("shares", 0) for s in codes] for p in profiles], dtype=float)
        self.belief = np.array([[np.log(self.price[j]) + p["belief_numeric"][s] for j, s in enumerate(codes)] for p in profiles])
        for name in ("info_weight", "social_weight", "risk", "loss_aversion", "trend", "max_leverage", "info_probability"):
            setattr(self, name, np.array([p[name] for p in profiles], dtype=float))
        self.source = np.array([p["information_source"] for p in profiles])
        ids = {u: i for i, u in enumerate(self.ids)}
        net = np.zeros((self.c.agents, self.c.agents))
        for e in snapshot["edges"]:
            # source influences target: matrix row is the listening investor.
            net[ids[str(e["target"])] , ids[str(e["source"])]] += e["weight"]
        self.network = normalize_network(net)
        self.snapshot_metadata = snapshot.get("metadata", {})

    def run(self, recorder: Optional[Callable] = None):
        c = self.c
        history = [self.price.copy()]
        market_rows, asset_rows = [], []
        max_cash_error = max_stock_error = max_debt_error = 0.0
        min_cash, min_dealer_cash, min_inventory = float(self.cash.min()), self.dealer_cash, float(self.dealer_inventory.min())
        cumulative_transmissions = 0
        initial = {"user_ids": self.ids, "asset_codes": self.codes, "cash": self.cash.tolist(), "debt": self.debt.tolist(),
                   "holdings": self.holdings.tolist(), "belief_log_value": self.belief.tolist(), "network": self.network.tolist(),
                   "information_source": self.source.tolist(), "risk": self.risk.tolist(), "loss_aversion": self.loss_aversion.tolist(),
                   "social_weight": self.social_weight.tolist(), "info_weight": self.info_weight.tolist(),
                   "info_probability": self.info_probability.tolist(), "trend": self.trend.tolist(),
                   "max_leverage": self.max_leverage.tolist(), "snapshot_metadata": self.snapshot_metadata,
                   "shock_target_agents": np.flatnonzero(self.target_agents).tolist(), "shock_asset_loadings": self.target_assets.tolist()}
        last_ret = np.zeros(c.assets)
        last_prevalence = 0.0
        for t in range(c.steps):
            active = self.control != "no_shock" and c.shock_size > 0
            age = t - c.shock_step
            severity = c.shock_size / 0.12 if active else 0.0
            if active and age == 0:
                if self.mechanism == "information":
                    self.fundamental -= c.shock_size * self.target_assets  # permanent level loss
                elif self.mechanism == "liquidity":
                    self.pending[self.target_agents] += 0.30 * severity * self.initial_equity[self.target_agents]
            # Dealer liquidity recovers only under an explicit duration assumption.
            stress = 0.0
            if active and age >= 0:
                stress = min(1.0, max(0.0, 1 - max(0, age - c.stress_duration + 1) / c.recovery_steps))
            depth = self.supply * c.depth_fraction
            if self.mechanism == "liquidity":
                depth = depth / (1 + 2 * severity * stress)
            equity = self.cash + self.holdings @ self.price - self.debt
            self.peak_wealth = np.maximum(self.peak_wealth, equity)
            drawdown = np.maximum(0, 1 - equity / np.maximum(self.peak_wealth, 1))
            # Pay external withdrawals and repay margin debt from idle cash first.
            withdrawal = np.minimum(self.cash, self.pending)
            self.cash -= withdrawal
            self.pending -= withdrawal
            self.withdrawn += float(withdrawal.sum())
            leverage_limit = self.max_leverage.copy()
            if self.mechanism == "deleveraging":
                leverage_limit = np.maximum(1.05, leverage_limit - 1.15 * severity * stress)
            equity = self.cash + self.holdings @ self.price - self.debt
            # Selling and repaying debt lowers assets while leaving equity unchanged.
            repayment_need = np.maximum(0, self.cash + self.holdings @ self.price - leverage_limit * np.maximum(equity, 0))
            repayment_need = np.minimum(repayment_need, self.debt)
            idle_repay = np.minimum(self.cash, repayment_need)
            self.cash -= idle_repay
            self.debt -= idle_repay
            self.lender_cash += float(idle_repay.sum())
            forced_need = self.pending + repayment_need - idle_repay
            peer_delta, peer_fear = social_update(self.belief.copy(), self.fear.copy(), self.network, self.social_weight)
            observes = self.observation_arrival[t] < self.info_probability
            private_delta = self.info_weight[:, None] * (self.fundamental[None, :] + self.observation_noise[t] - self.belief) * observes[:, None]
            new_belief = self.belief + private_delta + peer_delta
            public_fear = np.zeros(c.agents)
            if active and self.mechanism == "information" and 0 <= age < 3:
                public_fear = severity * 0.35 * observes
            rumor_seed = np.zeros(c.agents)
            if active and self.mechanism == "herding" and 0 <= age < 2:
                rumor_seed[self.target_agents] = min(0.9, 0.7 * severity)
                new_belief[self.target_agents] -= c.shock_size * self.target_assets
            exposure = self.social_weight * peer_fear
            new_fear = np.clip(c.emotion_decay * self.fear + 1.6 * self.loss_aversion * np.maximum(-last_ret.mean(), 0)
                               + 0.05 * drawdown + exposure * 0.6 + public_fear
                               + 0.45 * np.minimum(forced_need / np.maximum(self.initial_equity, 1), 1) + rumor_seed, 0, 1)
            if self.control == "random_emotion":
                new_fear = self.emotion_tape[t].copy()  # IID, independent of information/network
            if self.control == "simple_rule":
                new_fear[:] = 0
            newly = (new_fear >= c.panic_threshold) & (self.fear < c.panic_threshold)
            recovered = (new_fear < c.panic_threshold) & (self.fear >= c.panic_threshold)
            self.ever_panicked |= new_fear >= c.panic_threshold
            transmissions = int((newly & (peer_fear >= c.panic_threshold) & (self.social_weight > 0)).sum())
            cumulative_transmissions += transmissions  # exposure-associated, not causal attribution
            self.belief, self.fear = new_belief, new_fear
            gap = self.belief - np.log(self.price)[None, :]
            if self.control == "simple_rule":
                conviction = np.broadcast_to(-4 * last_ret, gap.shape)
            else:
                conviction = 5 * c.value_anchor * gap + self.trend[:, None] * last_ret * 3 - self.risk[:, None] * self.fear[:, None] * 0.55
            equity = self.cash + self.holdings @ self.price - self.debt
            budget = np.maximum(equity + self.debt, 0)
            target_value = self.preference * budget[:, None] * np.clip(0.88 + conviction, 0.05, 1.4)
            discretionary = 0.18 * (target_value / self.price - self.holdings)
            held_value = self.holdings @ self.price
            fraction = np.minimum(1, forced_need / np.maximum(held_value * (1 - c.agent_fee_bps / 10000), 1e-12))
            forced_orders = self.holdings * fraction[:, None]
            # A constrained account cannot simultaneously submit buy orders.
            orders = np.where(forced_need[:, None] > 1e-8, -np.maximum(forced_orders, np.maximum(-discretionary, 0)), discretionary)
            orders = np.maximum(orders, -self.holdings)
            # Conservative pre-quote buy budget using maximum possible quote rise.
            max_price = self.price * np.exp(c.impact + 0.02)
            buy_bill = np.maximum(orders, 0) @ max_price * (1 + c.agent_fee_bps / 10000)
            orders = np.where(orders > 0, orders * np.minimum(1, self.cash / np.maximum(buy_bill, 1e-12))[:, None], orders)
            imbalance = orders.sum(axis=0)
            quote_ret = c.impact * np.tanh(imbalance / depth) + self.market_noise[t]
            quote_ret = np.clip(quote_ret, -0.2, 0.2)  # numerical cap, NOT exchange price bands
            old_price = self.price.copy()
            self.price = self.price * np.exp(quote_ret)
            executed, forced_filled, cash, dinv, dcash, fees, bought, sold = clear_market(
                orders, forced_orders, self.holdings, self.cash, self.dealer_inventory,
                self.dealer_cash, self.price, c.agent_fee_bps / 10000)
            self.holdings += executed
            self.cash, self.dealer_inventory, self.dealer_cash = cash, dinv, dcash
            self.fee_sink += float(fees.sum())
            # Execute pending withdrawals and margin repayments with sale proceeds.
            paid = np.minimum(self.cash, self.pending)
            self.pending -= paid
            self.cash -= paid
            self.withdrawn += float(paid.sum())
            repay = np.minimum(np.minimum(self.cash, np.maximum(repayment_need - idle_repay, 0)), self.debt)
            self.cash -= repay
            self.debt -= repay
            self.lender_cash += float(repay.sum())
            ret = self.price / old_price - 1
            last_ret = ret.copy()
            sell_amount = sold.sum(axis=0) * self.price
            # One-sided matched volume: agent-agent cross counted once; dealer prints once.
            matched_shares = np.maximum(bought.sum(axis=0), sold.sum(axis=0))
            amount = matched_shares * self.price
            total_amount = float(amount.sum())
            seller_amount = sold @ self.price
            forced_amount = float((forced_filled * self.price).sum())
            prior_amounts = np.array([r["turnover_amount"] for r in market_rows[-20:]])
            if len(prior_amounts) == 20 and np.std(np.log1p(prior_amounts), ddof=1) > 1e-10:
                volume_z = float((np.log1p(total_amount) - np.mean(np.log1p(prior_amounts))) / np.std(np.log1p(prior_amounts), ddof=1))
            else:
                volume_z = None
            prevalence = float(np.mean(self.fear >= c.panic_threshold))
            disagreement = float(np.mean(np.std(gap, axis=0)))
            d_bin = ("low", "medium", "high")[int(disagreement >= c.disagreement_cuts[0]) + int(disagreement >= c.disagreement_cuts[1])]
            prior_forced = max((r["forced_sell_amount_latent"] for r in market_rows[-3:]), default=0)
            if prior_forced > 0.001 * float(self.supply @ self.price) and forced_amount < prior_forced * 0.5:
                latent_stage = "exhaustion"
            elif prevalence < 0.1 and forced_amount < 0.001 * float(self.supply @ self.price):
                latent_stage = "calm"
            elif prevalence > last_prevalence + 0.08:
                latent_stage = "onset" if last_prevalence < 0.1 else "propagation"
            elif prevalence < last_prevalence - 0.02:
                latent_stage = "recovery"
            else:
                latent_stage = "stress"
            last_prevalence = prevalence
            market_return = float(ret.mean())
            # A separate, causal market-data proxy never uses latent panic or event type.
            if market_return < -0.01:
                proxy_stage = "selloff"
            elif market_rows and market_rows[-1]["proxy_stage"] in ("selloff", "stabilizing") and market_return <= 0:
                proxy_stage = "stabilizing"
            elif market_rows and market_rows[-1]["proxy_stage"] in ("selloff", "stabilizing"):
                proxy_stage = "rebound"
            else:
                proxy_stage = "ordinary"
            recent_market = [r["market_return"] for r in market_rows[-19:]] + [market_return]
            row = {"step": t, "mechanism": self.mechanism, "control": self.control, "seed": c.seed,
                   "market_return": market_return, "breadth": float(np.mean(ret > 0)), "return_dispersion": float(ret.std()),
                   "market_vol20": float(np.std(recent_market, ddof=1)) if len(recent_market) == 20 else None,
                   "turnover_amount": total_amount, "volume_z20": volume_z,
                   "amihud": abs(market_return) / total_amount if total_amount > 0 else None,
                   "sell_concentration_assets": float(np.square(sell_amount / sell_amount.sum()).sum()) if sell_amount.sum() > 0 else None,
                   "seller_hhi_latent": float(np.square(seller_amount / seller_amount.sum()).sum()) if seller_amount.sum() > 0 else None,
                   "submitted_sell_imbalance": float(-(orders * self.price).sum() / max((np.abs(orders) * self.price).sum(), 1e-12)),
                   "forced_sell_amount_latent": forced_amount,
                   "forced_unfilled_amount_latent": float(((forced_orders - forced_filled) * self.price).sum()),
                   "fear_prevalence_latent": prevalence, "fear_new_share_latent": float(newly.mean()),
                   "fear_recovered_share_latent": float(recovered.mean()),
                   "diffusion_speed_latent": float(newly.mean() - recovered.mean()),
                   "ever_panicked_share_latent": float(self.ever_panicked.mean()),
                   "peer_associated_crossings_latent": transmissions, "belief_disagreement_latent": disagreement,
                   "disagreement_bin_latent": d_bin, "latent_stage": latent_stage, "proxy_stage": proxy_stage,
                   "withdrawals_cumulative": self.withdrawn, "debt_repaid_cumulative": self.lender_cash,
                   "pending_withdrawals": float(self.pending.sum()), "dealer_cash": self.dealer_cash,
                   "depth_multiplier": float((depth / (self.supply * c.depth_fraction)).mean())}
            market_rows.append(row)
            for j, code in enumerate(self.codes):
                asset_rows.append({"step": t, "asset": code, "seed": c.seed, "mechanism": self.mechanism,
                                   "control": self.control, "close": float(self.price[j]), "return": float(ret[j]),
                                   "volume": float(matched_shares[j]), "amount": float(amount[j]),
                                   "sell_amount": float(sell_amount[j]), "net_order_shares": float(imbalance[j]),
                                   "fundamental_latent": float(np.exp(self.fundamental[j]))})
            if recorder is not None:
                for i, user_id in enumerate(self.ids):
                    recorder({"step": t, "user_id": user_id, "belief_log_value": self.belief[i].tolist(),
                              "private_belief_delta": private_delta[i].tolist(), "social_belief_delta": peer_delta[i].tolist(),
                              "peer_fear_previous": float(peer_fear[i]), "fear": float(self.fear[i]),
                              "new_panic": bool(newly[i]), "recovered": bool(recovered[i]),
                              "cash": float(self.cash[i]), "debt": float(self.debt[i]), "holdings": self.holdings[i].tolist(),
                              "forced_need": float(forced_need[i]), "withdrawal_pending": float(self.pending[i]),
                              "submitted_shares": orders[i].tolist(), "executed_shares": executed[i].tolist(),
                              "forced_requested_shares": forced_orders[i].tolist(), "forced_filled_shares": forced_filled[i].tolist()})
            stock_error = float(np.max(np.abs(self.holdings.sum(axis=0) + self.dealer_inventory - self.supply)))
            cash_error = abs(float(self.cash.sum()) + self.dealer_cash + self.withdrawn + self.lender_cash + self.fee_sink - self.initial_cash)
            debt_error = abs(float(self.debt.sum()) + self.lender_cash - self.initial_debt)
            max_stock_error, max_cash_error, max_debt_error = max(max_stock_error, stock_error), max(max_cash_error, cash_error), max(max_debt_error, debt_error)
            min_cash, min_dealer_cash = min(min_cash, float(self.cash.min())), min(min_dealer_cash, self.dealer_cash)
            min_inventory = min(min_inventory, float(self.dealer_inventory.min()), float(self.holdings.min()))
            if min(min_cash, min_dealer_cash, min_inventory, float(self.debt.min())) < -1e-6:
                raise AssertionError("Cash, stock inventory or debt became negative")
            if not np.isfinite(self.price).all() or max_stock_error > 1e-6 or max_cash_error > 1e-5 or max_debt_error > 1e-5:
                raise AssertionError("Accounting or finite-price invariant failed")
            history.append(self.price.copy())
        return Result(c, self.mechanism, self.control, np.asarray(history), market_rows, asset_rows, initial,
                      {"max_cash_conservation_error": max_cash_error, "max_stock_conservation_error": max_stock_error,
                       "max_debt_conservation_error": max_debt_error, "min_agent_cash": min_cash,
                       "min_dealer_cash": min_dealer_cash, "min_inventory": min_inventory,
                       "exposure_associated_crossings": cumulative_transmissions})
