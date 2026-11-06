"""Explicit boundary to existing TwinMarket profiles, graphs and order schema.

No import of simulation.py or trading_agent.py: those load remote-model state.
Text belief is NEVER silently interpreted as fear or a numeric valuation.
"""
import copy
import json
import math


TRAITS = ("debt", "risk", "loss_aversion", "social_weight", "info_weight", "info_probability", "trend", "max_leverage", "information_source")


def from_twinmarket(profiles, user_graph, beliefs, belief_encoder, traits, prices, as_of):
    """Convert the objects available at process_user_input's boundary.

    belief_encoder(raw_belief, prices) must explicitly return one expected
    log-value gap per asset, e.g. log(subjective value / current price). traits
    is keyed by user ID, with all fields explicit, including debt=0 when assumed.
    A NetworkX graph or a list of source/target/weight dicts is accepted.
    """
    if not callable(belief_encoder) or not as_of:
        raise ValueError("An explicit belief encoder and snapshot as_of are required")
    raw_beliefs = {str(k): v for k, v in beliefs.items()}
    raw_traits = {str(k): v for k, v in traits.items()}
    converted = []
    for source in profiles:
        uid = str(source["user_id"])
        p = {"user_id": uid, "current_cash": source["current_cash"],
             "cur_positions": json.loads(source["cur_positions"]) if isinstance(source["cur_positions"], str) else copy.deepcopy(source["cur_positions"])}
        if uid not in raw_beliefs or uid not in raw_traits or any(name not in raw_traits[uid] for name in TRAITS):
            raise ValueError(f"Missing explicit belief/traits for {uid}")
        p.update({name: raw_traits[uid][name] for name in TRAITS})
        p["raw_belief"] = raw_beliefs[uid]
        p["belief_numeric"] = belief_encoder(raw_beliefs[uid], prices)
        converted.append(p)
    if isinstance(user_graph, list):
        edges = copy.deepcopy(user_graph)
    else:
        edges = []
        for u, v, data in user_graph.edges(data=True):
            edges.append({"source": str(u), "target": str(v), "weight": float(data.get("weight", 1))})
            if not user_graph.is_directed():
                edges.append({"source": str(v), "target": str(u), "weight": float(data.get("weight", 1))})
    result = {"prices": copy.deepcopy(prices), "profiles": converted, "edges": edges,
              "metadata": {"as_of": as_of, "belief_coding": "explicit user-supplied encoder",
                           "traits_provenance": "explicit external assumptions; not inferred from TwinMarket text",
                           "graph_warning": "TwinMarket similarity graph is not proof of observed social transmission"}}
    validate_snapshot(result)
    return result


def _finite(x, name, low=None, high=None):
    if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number")
    if low is not None and x < low or high is not None and x > high:
        raise ValueError(f"{name} outside supported range")


def validate_snapshot(snapshot):
    prices, profiles = snapshot["prices"], snapshot["profiles"]
    if len(prices) < 4 or len(profiles) < 4:
        raise ValueError("Snapshot needs at least four assets and four investors")
    if not snapshot.get("metadata", {}).get("as_of"):
        raise ValueError("Snapshot metadata.as_of is required")
    for code, price in prices.items():
        if not isinstance(code, str) or not code:
            raise ValueError("Asset codes must be strings")
        _finite(price, "price", 1e-12)
    ids = [str(p["user_id"]) for p in profiles]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate investor ID after string normalization")
    for p in profiles:
        for name in TRAITS:
            if name not in p:
                raise ValueError(f"Missing explicit {name}")
        for name in ("current_cash", "debt", "risk", "loss_aversion"):
            _finite(p[name], name, 0)
        _finite(p["max_leverage"], "max_leverage", 1.01)
        for name in ("social_weight", "info_weight", "info_probability"):
            _finite(p[name], name, 0, 1)
        if p["social_weight"] + p["info_weight"] > 1:
            raise ValueError("Belief-update weights must sum to <=1")
        _finite(p["trend"], "trend", -5, 5)
        if not isinstance(p["information_source"], (str, int)):
            raise ValueError("information_source must be a label")
        if set(p["belief_numeric"]) != set(prices):
            raise ValueError("Explicit numeric belief for every asset is required")
        for gap in p["belief_numeric"].values():
            _finite(gap, "belief gap", -2, 2)
        for code, pos in p["cur_positions"].items():
            if code not in prices:
                raise ValueError(f"Unknown held asset {code}")
            _finite(pos["shares"], "shares", 0)
        equity = p["current_cash"] + sum(prices[s] * pos["shares"] for s, pos in p["cur_positions"].items()) - p["debt"]
        if equity <= 0:
            raise ValueError("Initial equity must be positive")
    for edge in snapshot["edges"]:
        if str(edge["source"]) not in ids or str(edge["target"]) not in ids:
            raise ValueError("Unknown network endpoint")
        _finite(edge["weight"], "edge weight", 0)


def orders_to_twinmarket(user_ids, asset_codes, signed_shares, prices, lot_size=100):
    """Export *requests* to matching_engine decisions; no promised executions.

    Existing matcher fields: user_id, stock_code, direction, amount, target_price.
    Rounds down; run original holdings/cash/price-band checks before submission.
    """
    if type(lot_size) is not int or lot_size < 1:
        raise ValueError("lot_size must be a positive integer")
    if len(signed_shares) != len(user_ids) or len(prices) != len(asset_codes):
        raise ValueError("Order shape mismatch")
    if len(set(map(str, user_ids))) != len(user_ids) or len(set(asset_codes)) != len(asset_codes):
        raise ValueError("Duplicate investor or asset code")
    for price in prices:
        _finite(price, "target price", 1e-12)
    output = []
    for uid, orders in zip(user_ids, signed_shares):
        if len(orders) != len(asset_codes):
            raise ValueError("Order shape mismatch")
        for code, shares, price in zip(asset_codes, orders, prices):
            _finite(shares, "order shares")
            amount = int(abs(shares) // lot_size) * lot_size
            if amount:
                output.append({"user_id": str(uid), "stock_code": code, "direction": "buy" if shares > 0 else "sell",
                               "amount": amount, "target_price": float(price)})
    return output
