"""S0 main runs: the model's choice at the decision point, for every prompt a config asks for.

    python scripts/run_s0.py configs/s0_dev.yaml                # full run
    python scripts/run_s0.py configs/s0_dev.yaml --tasks 1      # one task only: a quick look
    python scripts/run_s0.py configs/s0_dev.yaml --tiny --tasks 1   # random 2-layer model: tests the code, no 16 GB download

Writes a run folder with results.jsonl (one line per prompt), prompts.jsonl (the exact prompt texts),
pairs.jsonl (do twin prompts line up?) and summary.md.
"""
import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detour import content, grid, model as M, report  # noqa: E402
from detour.config import config_files, load_config  # noqa: E402
from detour.prompts import build, compare_ids, tokenize  # noqa: E402
from detour.score import score  # noqa: E402
from lab import Run, snapshot_files  # noqa: E402


def twin_pairs(prompts):
    """Allow/forbid twins and blocked/not-blocked twins: same length, and differing only inside the swapped part?"""
    out = []
    for key, p in prompts.items():
        s = p.setting
        checks = []
        if s.rule_type == "forbid":
            checks.append(("allow vs forbid", s.key(rule_type="allow"), "rule"))
        if s.rule_type == "implicit_forbid":
            checks.append(("implicit allow vs forbid", s.key(rule_type="implicit_allow"), "mirror_note"))
        if s.blocked:
            checks.append(("blocked vs not blocked", s.key(blocked=False), "route_a_reply"))
        for kind, twin_key, part in checks:
            twin = prompts.get(twin_key)
            if twin is None:
                continue
            c = compare_ids(twin.ids, p.ids)
            lo, hi = p.spans.get(part, (0, 0))
            inside = c["same_length"] and (c["first_diff"] is None or (lo <= c["first_diff"] and c["last_diff"] < hi))
            out.append({"kind": kind, "key": key, "twin": twin_key, "inside_swap": inside, **c})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tasks", type=int, help="use only the first N tasks of the split")
    ap.add_argument("--tiny", action="store_true", help="random 2-layer model (code test only)")
    ap.add_argument("--tokenizer", help="tokenizer path or name (default: the config's model)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    settings = grid.settings(cfg)
    if args.tasks:
        keep = {t["id"] for t in content.tasks(cfg["split"])[: args.tasks]}
        settings = [s for s in settings if s.task_id in keep]
    if args.tiny:
        cfg = cfg | {"name": cfg["name"] + "_TINY_TEST", "model": "tiny random (code test)"}
        tok = M.load_tokenizer(args.tokenizer or load_config(args.config)["model"])
        mdl, device = M.tiny_random(tok), "cpu"
    else:
        mdl, tok, device = M.load(cfg["model"], cfg.get("dtype", "bfloat16"), cfg.get("device", "auto"),
                                  tokenizer=args.tokenizer)

    run = Run(cfg["name"], cfg, code_files=snapshot_files(__file__, *config_files(args.config)))
    run.add_meta(model_info=M.describe(mdl, device), n_prompts=len(settings))
    print(f"{len(settings)} prompts on {device}")

    prompts, rows, seen = {}, [], set()
    start = time.time()
    for i, s in enumerate(settings, 1):
        p = tokenize(tok, build(s))
        key = s.key()
        prompts[key] = p
        res = score(mdl, tok, p, device)
        style = content.rule_set(s.rule_set_id)["style"] if s.rule_set_id else None
        row = {**asdict(s), "key": key, "twin_allow": s.key(rule_type="allow"), "rule_style": style,
               "rule_text": p.rule_text, "prompt_sha": p.sha, "n_tokens": len(p.ids),
               "positions": p.positions, "spans": p.spans, **res}
        rows.append(row)
        run.log(row)
        if p.sha not in seen:
            seen.add(p.sha)
            run.log_to("prompts.jsonl", {"prompt_sha": p.sha, "text": p.text, "ids": p.ids})
        if i % 25 == 0 or i == len(settings):
            rate = (time.time() - start) / i
            print(f"[{i}/{len(settings)}] {rate:.1f} s/prompt, about {rate * (len(settings) - i) / 60:.0f} min left")

    pairs = twin_pairs(prompts)
    for pr in pairs:
        run.log_to("pairs.jsonl", pr)
    run.finish(report.s0_summary(cfg, rows, pairs))


if __name__ == "__main__":
    main()
