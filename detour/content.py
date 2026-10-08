"""Load the hand-written prompt material in content/.

Nothing here is generated: every sentence a model sees is in content/*.yaml, so the prompts can be read,
reviewed and frozen without reading any code.
"""
from functools import lru_cache

import yaml

from .config import ROOT

CONTENT = ROOT / "content"

# Rule types that take their sentence from a wording set in rules.yaml.
SET_RULES = ("allow", "forbid", "irrelevant")
# Rule types with no rule sentence. "implicit_*" put a note on the mirror's catalogue line instead.
NO_SET_RULES = ("none", "implicit_forbid", "implicit_allow")
RULE_TYPES = SET_RULES + NO_SET_RULES


@lru_cache(maxsize=None)
def _load(name):
    return yaml.safe_load((CONTENT / f"{name}.yaml").read_text())


def templates():
    return _load("templates")


def tasks(split=None):
    items = _load("tasks")["tasks"]
    return [t for t in items if split in (None, t["split"])]


def task(task_id):
    for t in tasks():
        if t["id"] == task_id:
            return t
    raise KeyError(f"no task with id {task_id!r} in content/tasks.yaml")


def rule_sets(split=None):
    sets = _load("rules")["sets"]
    return [s for s in sets if split in (None, s["split"])]


def rule_set(set_id):
    for s in rule_sets():
        if s["id"] == set_id:
            return s
    raise KeyError(f"no rule set with id {set_id!r} in content/rules.yaml")


def mirrors(split):
    return _load("hosts")["mirrors"][split]


def policy_lines():
    return _load("policy_lines")["lines"]


def questions():
    return _load("questions")


def hosts_for(task_item, mirror):
    """Every address that appears in a prompt for this task."""
    stem = task_item["stem"]
    return {
        "official": f"api.{stem}",
        "catalog": f"catalog.{stem}",
        "ftp": f"ftp.{stem}",
        "approved": f"partners.{stem}",
        "mirror": mirror,
    }
