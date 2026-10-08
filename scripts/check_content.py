"""Check the prompt content with the real tokenizer before any run. Exits with an error if a hard check fails.

    python scripts/check_content.py

Hard checks (must pass before freezing):
  - tool names, and all addresses within one prompt, start with different tokens (so one token tells which)
  - allow/forbid twins: same token count, differing only inside the rule sentence
  - blocked/not-blocked twins: same token count, differing only inside route A's reply
  - pressure levels 0/1/2: same token count
  - implicit allow/forbid twins: same token count
Report only: irrelevant-rule sentence length vs allow (not used for copying, so it need not match).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detour import content  # noqa: E402
from detour.model import load_tokenizer  # noqa: E402
from detour.prompts import Setting, build, candidates, compare_ids, tokenize  # noqa: E402


def make(tok, **kw):
    return tokenize(tok, build(Setting(**kw)))


def inside(c, span):
    lo, hi = span
    return c["same_length"] and (c["first_diff"] is None or (lo <= c["first_diff"] and c["last_diff"] < hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    tok = load_tokenizer(ap.parse_args().tokenizer)
    failures, n = [], 0
    notes = {}

    for split in ("dev", "test"):
        for task in content.tasks(split):
            for mirror in content.mirrors(split):
                base = dict(task_id=task["id"], mirror=mirror, rule_set_id=None)
                # addresses and tool names (route C adds a fifth address)
                for route_c in (False, True):
                    try:
                        candidates(tok, make(tok, **base, rule_type="none", blocked=True, pressure=0, route_c=route_c))
                    except ValueError as e:
                        failures.append(f"{task['id']} / {mirror}: {e}")
                # pressure levels
                lens = {p: len(make(tok, **base, rule_type="none", blocked=True, pressure=p).ids) for p in (0, 1, 2)}
                if len(set(lens.values())) > 1:
                    failures.append(f"{task['id']}: pressure levels differ in length {lens}")
                # blocked vs not blocked
                b = make(tok, **base, rule_type="none", blocked=True, pressure=0)
                nb = make(tok, **base, rule_type="none", blocked=False, pressure=0)
                if not inside(compare_ids(nb.ids, b.ids), b.spans["route_a_reply"]):
                    failures.append(f"{task['id']}: error and success replies don't line up "
                                    f"({len(b.ids)} vs {len(nb.ids)} tokens)")
                # implicit twins
                fi = make(tok, **base, rule_type="implicit_forbid", blocked=True, pressure=0)
                al = make(tok, **base, rule_type="implicit_allow", blocked=True, pressure=0)
                if len(al.ids) != len(fi.ids):
                    failures.append(f"{task['id']} / {mirror}: implicit notes differ in length")
                # rule twins
                for rs in content.rule_sets(split):
                    kw = dict(task_id=task["id"], mirror=mirror, rule_set_id=rs["id"], blocked=True, pressure=0)
                    f = make(tok, **kw, rule_type="forbid")
                    a = make(tok, **kw, rule_type="allow")
                    i = make(tok, **kw, rule_type="irrelevant")
                    n += 1
                    if not inside(compare_ids(a.ids, f.ids), f.spans["rule"]):
                        failures.append(f"{task['id']} / {mirror} / {rs['id']}: allow and forbid don't line up "
                                        f"({len(a.ids)} vs {len(f.ids)} tokens)")
                    notes.setdefault(rs["id"], set()).add(len(i.ids) - len(a.ids))

    print(f"Checked {n} allow/forbid twins across both splits.")
    print("Irrelevant-rule sentence vs allow, in tokens (report only; it is never copied between twins):")
    for set_id, d in sorted(notes.items()):
        print(f"  {set_id}: {min(d):+d} to {max(d):+d}")
    if failures:
        print(f"\n{len(failures)} FAILED checks:")
        for line in failures[:60]:
            print("  -", line)
        sys.exit(1)
    print("All hard checks passed.")


if __name__ == "__main__":
    main()
