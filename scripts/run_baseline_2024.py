"""Run a deliberately simple, leakage-safe 5-day momentum baseline on 2024."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load_panel import load_panel
from src.metrics.diagnostics import monthly_score_diagnostics
from src.metrics.official import score_official
from src.validation.splits import get_split


def _load_official_evaluator():
    # The supplied scorer uses loguru solely for ``logger.info``.  Keep its metric
    # code untouched even in a minimal environment where that optional logger is
    # absent.
    if importlib.util.find_spec("loguru") is None:
        shim = types.ModuleType("loguru")
        shim.logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
        sys.modules["loguru"] = shim
    spec = importlib.util.spec_from_file_location("competition_evaluate", ROOT / "赛题五" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.evaluate


def main() -> None:
    output_dir = ROOT / "artifacts" / "baseline_2024"
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["ts_code", "trade_date", "close", "flag_limit_up", "y_ret_1d"]
    panel = load_panel("train", columns=columns, add_row_id=False, sort_for_features=True)
    # At decision-day close, both close[t] and historical close[t-5] are known.
    panel["mom_5d"] = panel.groupby("ts_code", sort=False)["close"].pct_change(5, fill_method=None)
    split = get_split("oos_2024")
    train_mask, valid_mask = split.masks(panel)
    assert panel.loc[train_mask, "trade_date"].max() < split.valid_start
    valid = panel.loc[valid_mask].copy()
    valid["pred"] = valid["mom_5d"].fillna(0.0)
    pred = valid[["ts_code", "trade_date", "pred"]]
    truth = valid[["ts_code", "trade_date", "y_ret_1d"]]
    x = valid[["ts_code", "trade_date", "flag_limit_up"]]
    local = score_official(pred, truth, x, return_details=True)
    pred.to_csv(output_dir / "submission_2024.csv", index=False)
    local_details = local.pop("details")
    for name, frame in local_details.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    monthly_score_diagnostics(local_details).to_csv(output_dir / "monthly_diagnostics.csv", index=False)
    (output_dir / "local_metrics.json").write_text(json.dumps(local, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # The unchanged official script reads files by competition names.  Create a
    # disposable validation fixture under artifacts, then compare every scalar.
    fixture = output_dir / "evaluate_input"
    fixture.mkdir(exist_ok=True)
    x.to_csv(fixture / "测试集_X.csv", index=False)
    truth.to_csv(fixture / "测试集_Y.csv", index=False)
    official = _load_official_evaluator()(str(output_dir / "submission_2024.csv"), str(fixture))
    differences = {key: float(local[key] - official[key]) for key in local}
    if any(abs(value) > 1e-12 for value in differences.values()):
        raise AssertionError(f"local scorer diverges from official scorer: {differences}")
    (output_dir / "official_comparison.json").write_text(json.dumps({"official": official, "local_minus_official": differences}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(local, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
