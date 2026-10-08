"""S0 question checks ("forks"): does the model know route B is forbidden, that it would work,
and what a compliant agent should do? Run separately from the main runs; never mixed into them.

    python scripts/run_checks.py configs/checks_dev.yaml
    python scripts/run_checks.py configs/checks_dev.yaml --tasks 1 --tiny    # code test
"""
import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detour import checks, content, grid, model as M, report  # noqa: E402
from detour.config import config_files, load_config  # noqa: E402
from detour.prompts import build, tokenize  # noqa: E402
from lab import Run, snapshot_files  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tasks", type=int)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--tokenizer")
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
    print(f"{len(settings)} prompts x about 10 questions on {device}")

    rows = []
    for i, s in enumerate(settings, 1):
        p = tokenize(tok, build(s))
        for r in checks.run_all(mdl, tok, p, device, seed=cfg.get("seed", 0)):
            row = {**asdict(s), "prompt_sha": p.sha, **r}
            rows.append(row)
            run.log(row)
        if i % 10 == 0 or i == len(settings):
            print(f"[{i}/{len(settings)}]")
    run.finish(report.checks_summary(cfg, rows))


if __name__ == "__main__":
    main()
