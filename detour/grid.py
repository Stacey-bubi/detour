"""List every prompt setting a config asks for.

For each task in the split and each name draw (mirror address + tool order), every combination of
rule type x wording set x blocked x pressure. Twins (allow vs forbid, blocked vs not blocked) share the
same task, draw and wording set, so they differ in exactly one sentence.
"""
from . import content
from .prompts import Setting, draw_names


def settings(cfg):
    split, seed, grid = cfg["split"], cfg.get("seed", 0), cfg["grid"]
    opts = cfg.get("options", {})
    out = []
    for task in content.tasks(split):
        for draw in range(grid.get("name_draws", 1)):
            mirror, order = draw_names(task["id"], split, draw, seed)
            for rule_type in grid["rule_types"]:
                if rule_type not in content.RULE_TYPES:
                    raise ValueError(f"unknown rule type {rule_type!r}; known: {content.RULE_TYPES}")
                set_ids = ([s["id"] for s in content.rule_sets(split)]
                           if rule_type in content.SET_RULES else [None])
                for set_id in set_ids:
                    for blocked in grid["blocked"]:
                        for pressure in grid["pressure"]:
                            out.append(Setting(task_id=task["id"], rule_type=rule_type, rule_set_id=set_id,
                                               blocked=blocked, pressure=pressure, mirror=mirror,
                                               tool_order=order, draw=draw, **opts))
    return out
