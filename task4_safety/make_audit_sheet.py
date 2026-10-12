from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


POLICIES = ["sft", "dpo", "ppo", "grpo"]


def main():
    """Fixed 60-ID audit subset (30 SAFE + 30 UNSAFE, course seed) and a blind labelling sheet.

    manual_audit_ids.csv  - the fixed IDs (course function, unchanged)
    manual_audit_sheet.csv - one row per UNIQUE (xstest_id, response) among the four policies for
                             those IDs (identical responses are labelled once and the label is
                             applied to every policy that produced it), shuffled with the course
                             seed, with NO AI labels. Fill `manual_label` with one of
                             SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS.
    Re-running never overwrites a sheet that already has labels.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    sheet = outdir / "manual_audit_sheet.csv"
    if sheet.exists() and pd.read_csv(sheet)["manual_label"].fillna("").astype(str).str.strip().ne("").any():
        print("Labelled sheet already exists, not overwriting:", sheet)
        return
    idset = set(ids)
    items = {}
    for name in POLICIES:
        p = outdir / f"generated_{name}.jsonl"
        if not p.exists():
            print(f"  (no generations for {name}; sheet will not cover it)")
            continue
        for r in read_jsonl(p):
            if int(r["xstest_id"]) not in idset:
                continue
            key = (int(r["xstest_id"]), r["response"].strip())
            it = items.setdefault(key, {"xstest_id": int(r["xstest_id"]), "type": r["type"],
                                        "benchmark_class": r["benchmark_class"], "prompt": r["prompt"],
                                        "response": r["response"], "policies": []})
            it["policies"].append(name)
    df = pd.DataFrame(items.values())
    df["policies"] = df["policies"].map(lambda x: ";".join(x))
    df = df.sample(frac=1.0, random_state=int(cfg["seed"])).reset_index(drop=True)
    df.insert(0, "item_id", range(1, len(df) + 1))
    df["manual_label"] = ""
    df["notes"] = ""
    df.to_csv(sheet, index=False)
    print(f"Wrote blind audit sheet with {len(df)} unique responses for {len(ids)} IDs:", sheet)
    print("Label it without looking at judged_*.jsonl, then run task4_safety.evaluate_safety.")


if __name__ == "__main__":
    main()
