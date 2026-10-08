"""Print one prompt exactly as the model sees it, with the parts and the read positions marked.

    python scripts/show_prompt.py                                   # rainfall, forbid, blocked, pressure 0
    python scripts/show_prompt.py --task ferry --rule allow --set c3 --pressure 2
    python scripts/show_prompt.py --rule none --not-blocked
    python scripts/show_prompt.py --tokens                          # also list every token with its index

Use this before every run you change: it is the quickest way to catch a template or wording mistake.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detour import content  # noqa: E402
from detour.model import load_tokenizer  # noqa: E402
from detour.prompts import CALL_OPEN, CALL_TO_ADDRESS, TOOL_NAMES, Setting, build, candidates, tokenize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="rainfall")
    ap.add_argument("--rule", default="forbid", choices=content.RULE_TYPES)
    ap.add_argument("--set", default="c1", help="wording set id from content/rules.yaml")
    ap.add_argument("--not-blocked", action="store_true")
    ap.add_argument("--pressure", type=int, default=0)
    ap.add_argument("--mirror", default=None, help="mirror address (default: first in the task's split pool)")
    ap.add_argument("--placement", default="system", choices=["system", "user"])
    ap.add_argument("--policy-lines", type=int, default=0)
    ap.add_argument("--priority", action="store_true")
    ap.add_argument("--route-c", action="store_true")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--tokens", action="store_true")
    a = ap.parse_args()

    task = content.task(a.task)
    s = Setting(task_id=a.task, rule_type=a.rule, rule_set_id=a.set if a.rule in content.SET_RULES else None,
                blocked=not a.not_blocked, pressure=a.pressure, mirror=a.mirror or content.mirrors(task["split"])[0],
                tool_order=TOOL_NAMES, placement=a.placement, policy_lines=a.policy_lines,
                priority=a.priority, route_c=a.route_c)
    tok = load_tokenizer(a.tokenizer)
    p = tokenize(tok, build(s))

    print(p.text + CALL_OPEN + CALL_TO_ADDRESS)
    print("\n" + "=" * 80)
    print(f"{len(p.ids)} tokens. Read positions (the token whose next-token prediction we read):")
    for name, i in p.positions.items():
        print(f"  {name:11} index {i:5}  token {tok.decode([p.ids[i]])!r}")
    print("Parts (token index range):")
    for name, (lo, hi) in p.spans.items():
        print(f"  {name:22} {lo:5}-{hi:5}  {tok.decode(p.ids[lo:hi])[:70]!r}")
    cand = candidates(tok, p)
    print("Options read at 'tool':    ", {k: tok.decode([v]) for k, v in cand["tool"].items()})
    print("Options read at 'address': ", {k: tok.decode([v]) for k, v in cand["address"].items()})
    if a.tokens:
        print("\nAll tokens:")
        for i, t in enumerate(p.ids):
            print(f"{i:5} {tok.decode([t])!r}")


if __name__ == "__main__":
    main()
