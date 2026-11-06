"""Run a paired synthetic experiment matrix with only Python and NumPy."""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from .config import Config, CONTROLS, MECHANISMS
from .engine import Simulation
from .evaluation import conditional_targets, cost_adjust, paired_contrasts, summarize


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_digest():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def run_matrix(config, seeds, output, mechanisms=MECHANISMS, controls=CONTROLS, logs="example", snapshot=None):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct seeds")
    if not mechanisms or not controls or len(set(mechanisms)) != len(mechanisms) or len(set(controls)) != len(controls):
        raise ValueError("Provide distinct nonempty mechanisms and controls")
    if logs not in ("none", "example", "all"):
        raise ValueError("Unknown log setting")
    # Never mix stale runs or overwrite a previous result directory.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip() or None
    except OSError:
        commit = None
    manifest = {"evidence_level": "UNCALIBRATED_SYNTHETIC_MECHANISM_EXPERIMENT",
                "config": config.to_dict(), "seeds": list(seeds), "mechanisms": list(mechanisms), "controls": list(controls),
                "logs": logs, "python": platform.python_version(), "numpy": np.__version__,
                "source_sha256": source_digest(), "git_head": commit,
                "git_head_note": "Local checkout HEAD may predate uncommitted experiment files; source_sha256 identifies executed Python sources",
                "twinmarket_interface_baseline": "de5f2446fcba0d0aba533a6adaede034160e29b4",
                "snapshot_sha256": hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest() if snapshot else None,
                "units": "one step = one stylized trading day; not calibrated to seconds or Tick events",
                "execution": "signal close t; enter close t+1; exit close t+1+h; independent overlapping cohorts",
                "costs": "constant bps per notional leg; L/S gross exposure 2; no borrow fees/fill constraints",
                "uncertainty": "bootstrap independent seed means, NOT overlapping days; descriptive Monte Carlo intervals",
                "limitations": ["not an A-share exchange replica", "fractional shares and synthetic dealer quotes",
                                "no real-data calibration", "no realized portfolio or tradable alpha claim",
                                "latent stage/disagreement unavailable from price data alone", "simple cost stress test only"]}
    write_json(output / "manifest.json", manifest)
    all_market, all_assets, all_targets, audits, initial = [], [], [], [], []
    for mechanism in mechanisms:
        for control in controls:
            for seed in seeds:
                path = output / f"agents_{mechanism}_{control}_{seed}.jsonl"
                record = logs == "all" or logs == "example" and control == "baseline" and seed == seeds[0]
                stream = path.open("w", encoding="utf-8") if record else None
                def recorder(row):
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                try:
                    result = Simulation(replace(config, seed=seed), mechanism, control, snapshot).run(recorder if stream else None)
                finally:
                    if stream:
                        stream.close()
                all_market.extend(result.market)
                all_assets.extend(result.assets)
                targets = conditional_targets(result)
                all_targets.extend(targets)
                audits.append({"mechanism": mechanism, "control": control, "seed": seed, **result.audit})
                if record:
                    initial.append({"mechanism": mechanism, "control": control, "seed": seed, **result.initial})
        print(f"completed {mechanism}: {len(controls) * len(seeds)} runs", flush=True)
    summary = summarize(all_targets, config.cost_bps_per_side)
    paired = paired_contrasts(all_targets, config.cost_bps_per_side)
    write_csv(output / "market.csv", all_market)
    write_csv(output / "assets.csv", all_assets)
    write_csv(output / "conditional_targets.csv", all_targets)
    write_csv(output / "conditional_summary.csv", summary)
    write_csv(output / "paired_effects.csv", paired)
    write_csv(output / "accounting_audit.csv", audits)
    fixed = [cost_adjust(row, cost) for row in all_targets if row["fixed_event_signal"] for cost in config.cost_bps_per_side]
    write_csv(output / "fixed_event_targets.csv", fixed)
    write_json(output / "initial_states.json", initial)
    report = ["# 合成机制实验运行记录", "", "**证据等级：未经真实数据校准的合成模拟。以下收益不代表现实可交易 Alpha。**", "",
              f"运行 {len(audits)} 组；{config.agents} 代理 × {config.assets} 资产 × {config.steps} 日；种子 {list(seeds)}。",
              f"全部运行现金/股票/债务守恒通过；最大现金误差 {max(a['max_cash_conservation_error'] for a in audits):.3g}。", "",
              f"信号为过去 {config.lookback} 日收益排序的两端组合，每端包含 floor(K/3) 个资产。",
              "收盘 t 形成信号，t+1 收盘进入，t+1+h 收盘退出。表内为独立事件组合条件收益，不是连续账户回测。", "",
              "## 固定事件时点的基准组", "", "各情形统一使用冲击后第 2 日信号；不同情形的实际跌幅没有事后配平。", "",
              "| 机制 | 持有期 | 单边成本(bps) | 种子数 | 理论多空净收益均值 | 多头超额净收益代理均值 |", "|---|---:|---:|---:|---:|---:|"]
    for mechanism in mechanisms:
        for h in config.horizons:
            rows = [r for r in fixed if r["mechanism"] == mechanism and r["control"] == "baseline" and r["horizon"] == h and r["cost_bps_per_side"] == config.cost_bps_per_side[min(1, len(config.cost_bps_per_side)-1)]]
            if rows:
                report.append(f"| {mechanism} | {h} | {rows[0]['cost_bps_per_side']:g} | {len(rows)} | {np.mean([r['spread_net_theoretical'] for r in rows]):.3%} | {np.mean([r['long_only_excess_net_proxy'] for r in rows]):.3%} |")
    report += ["", "conditional_summary.csv 同时按机制、对照、阶段、分歧程度、持有期和成本分组。",
               "阶段与分歧会随干预变化，分组差异含样本构成变化；因果模拟对照请看固定时点 paired_effects.csv。",
               "多空不可默认在 A 股执行；多头超额是相对毛市场基准扣固定成本的代理，未模拟涨跌停、排队、冲击成本或借券。",
               "真实校准与样本外验证方案见 README.md。重跑时请使用全新输出目录。", ""]
    (output / "RUN_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    manifest["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    manifest["completed_runs"] = len(audits)
    manifest["completed"] = True
    write_json(output / "manifest.json", manifest)
    return {"output": str(output.resolve()), "runs": len(audits), "seconds": manifest["elapsed_seconds"], "source_sha256": manifest["source_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New, nonexistent output directory")
    parser.add_argument("--config", type=Path, help="JSON overrides for Config")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--mechanisms", nargs="+", choices=MECHANISMS, default=list(MECHANISMS))
    parser.add_argument("--controls", nargs="+", choices=CONTROLS, default=list(CONTROLS))
    parser.add_argument("--logs", choices=("none", "example", "all"), default="example")
    parser.add_argument("--value-anchor", type=float)
    parser.add_argument("--recovery-steps", type=int)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    for name in ("horizons", "cost_bps_per_side", "disagreement_cuts"):
        if name in settings:
            settings[name] = tuple(settings[name])
    if args.value_anchor is not None:
        settings["value_anchor"] = args.value_anchor
    if args.recovery_steps is not None:
        settings["recovery_steps"] = args.recovery_steps
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8")) if args.snapshot else None
    if snapshot:
        settings["agents"], settings["assets"] = len(snapshot["profiles"]), len(snapshot["prices"])
    print(json.dumps(run_matrix(Config(**settings), args.seeds, args.output, args.mechanisms, args.controls, args.logs, snapshot), ensure_ascii=False))


if __name__ == "__main__":
    main()
