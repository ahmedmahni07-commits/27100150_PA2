"""Count how the DPO length-handling rule touched each fixed file (no GPU needed).

Rule used for every Task 1 training and evaluation run (course patch of 3 Oct + filtering):
  * prompt alone >= max_sequence_length tokens  -> pair excluded (task1_dpo.train.filter_fitting_rows)
  * otherwise the prompt is kept whole and any response that does not fit is right-truncated
    (keeping EOS) by common.data.encode_prompt_response.
Writes results/task1_dpo/truncation_stats.json, broken down by length stratum where available.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from common.data import chat_prompt_ids, load_yaml, preference_responses, prompt_messages_from_preference, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tok = load_tokenizer(cfg["base_model"])
    L = int(cfg["max_sequence_length"])
    out = {"max_sequence_length": L, "files": {}}
    for key in ["dpo_standard_train", "dpo_standard_eval", "dpo_length_train", "dpo_length_eval"]:
        by = defaultdict(Counter)
        for r in read_jsonl(cfg["paths"][key]):
            s = r.get("length_stratum", "all")
            c = by[s]
            c["total"] += 1
            p = len(chat_prompt_ids(tok, prompt_messages_from_preference(r)))
            if p >= L:
                c["excluded_prompt_too_long"] += 1
                continue
            c["kept"] += 1
            yc, yr = preference_responses(r)
            tc = p + len(tok(yc, add_special_tokens=False)["input_ids"]) + 1 > L
            tr = p + len(tok(yr, add_special_tokens=False)["input_ids"]) + 1 > L
            c["kept_with_any_truncation"] += int(tc or tr)
            c["only_chosen_truncated"] += int(tc and not tr)
            c["only_rejected_truncated"] += int(tr and not tc)
            c["both_truncated"] += int(tc and tr)
        total = Counter()
        for c in by.values():
            total.update(c)
        out["files"][key] = {"path": cfg["paths"][key], "overall": dict(total),
                             **({"by_stratum": {k: dict(v) for k, v in by.items()}} if len(by) > 1 else {})}
        print(key, dict(total))
    save_json(repo_path(cfg["results_dir"]) / "truncation_stats.json", out)


if __name__ == "__main__":
    main()
