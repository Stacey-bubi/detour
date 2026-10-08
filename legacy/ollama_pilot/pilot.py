"""Ollama pilot: does the model take route B when the rule forbids it?

Usage:
    python legacy/ollama_pilot/pilot.py legacy/ollama_pilot/pilot_14b.yaml
    python legacy/ollama_pilot/pilot.py legacy/ollama_pilot/pilot_14b.yaml --samples 5      # quick dry run

All settings live in the config file!
"""
import argparse
import collections
import itertools
import time

import requests
import yaml

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for lab.py

from lab import Run  # noqa: E402

OLLAMA = "http://localhost:11434"
EXITS = ("ask_user", "report_blocked")


def make_tools(cfg, use_exits):
    tools = []
    for t in cfg["tools"]:
        if not use_exits and t["name"] in EXITS:
            continue
        props = {p: {"type": kind} for p, kind in t["parameters"].items()}
        tools.append({"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": {"type": "object", "properties": props}}})
    return tools


def make_messages(cfg, rule, placement, blocked, pressure):
    rule_text = cfg["rules"][rule]
    system = cfg["system_role"] + (" " + rule_text if placement == "system" else "")
    user = (rule_text + " " if placement == "user" else "") + cfg["task"] + cfg["pressure"][pressure]
    result = cfg["route_a_result"]["blocked" if blocked else "not_blocked"]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": "", "tool_calls": [{"function": cfg["route_a_call"]}]},
        {"role": "tool", "content": result},
    ]


def ask(cfg, messages, tools, seed):
    body = {"model": cfg["model"], "messages": messages, "tools": tools, "think": cfg["think"],
            "stream": False, "options": {"temperature": cfg["temperature"], "seed": seed}}
    r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=600).json()
    if "error" in r:
        raise RuntimeError(f"Ollama error for {cfg['model']}: {r['error']}")
    msg = r["message"]
    calls = msg.get("tool_calls") or []
    if calls:
        return calls[0]["function"]["name"], calls[0]["function"].get("arguments"), msg.get("content", "")
    return "plain text", None, msg.get("content", "")


def model_digest(name):
    for m in requests.get(f"{OLLAMA}/api/tags").json().get("models", []):
        if m["name"] == name:
            return m.get("digest")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--samples", type=int, help="override samples per setting (for a quick dry run)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.samples:
        cfg["samples"] = args.samples

    run = Run(cfg["name"], cfg, code_files=[__file__, "lab.py", args.config])
    run.add_meta(model_digest=model_digest(cfg["model"]))
    grid = cfg["grid"]

    # 1. Comprehension check: does the model know whether it may use route B?
    question = cfg["comprehension_question"]
    for rule, placement in itertools.product(grid["rules"], grid["placements"]):
        msgs = make_messages(cfg, rule, placement, True, 0) + [{"role": "user", "content": question}]
        for i in range(cfg["comprehension_samples"]):
            _, _, text = ask(cfg, msgs, [], seed=i)
            run.log({"kind": "comprehension", "rule": rule, "placement": placement,
                     "seed": i, "answer": text.strip()[:200]})

    # 2. Main grid
    settings = list(itertools.product(grid["rules"], grid["placements"], grid["blocked"],
                                      grid["pressure"], grid["exits"]))
    print(f"{len(settings)} settings x {cfg['samples']} samples = {len(settings) * cfg['samples']} calls")
    counts = {}
    start = time.time()
    for k, (rule, placement, blocked, pressure, exits) in enumerate(settings, 1):
        msgs = make_messages(cfg, rule, placement, blocked, pressure)
        tools = make_tools(cfg, exits)
        c = collections.Counter()
        for i in range(cfg["samples"]):
            choice, arguments, text = ask(cfg, msgs, tools, seed=i)
            c[choice] += 1
            run.log({"kind": "choice", "rule": rule, "placement": placement, "blocked": blocked,
                     "pressure": pressure, "exits": exits, "seed": i, "choice": choice,
                     "arguments": arguments, "text": text.strip()[:200]})
        counts[(rule, placement, blocked, pressure, exits)] = c
        mins = (time.time() - start) / 60
        print(f"[{k}/{len(settings)} | {mins:.0f} min] {rule} {placement} blocked={blocked} "
              f"p={pressure} exits={exits}: {dict(c)}")

    run.finish(summary(cfg, counts))


def summary(cfg, counts):
    b = cfg["route_b"]
    lines = [f"# {cfg['name']}", "",
             f"Model `{cfg['model']}`, think={cfg['think']}, temperature={cfg['temperature']}, "
             f"{cfg['samples']} samples per setting.", "",
             "| rule | placement | blocked | pressure | exits | route B | all choices |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for (rule, placement, blocked, pressure, exits), c in counts.items():
        n = sum(c.values())
        lines.append(f"| {rule} | {placement} | {blocked} | {pressure} | {exits} | "
                     f"{c[b]}/{n} | {dict(c)} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
