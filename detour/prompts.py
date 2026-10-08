"""Build one prompt from its parts, and find the token positions we read.

The conversation every prompt has, in this order:

  SYSTEM     role line + rule                 (rule swapped: allow / forbid / irrelevant / none)
             + tool definitions (added by Qwen3's chat template after the system text)
  USER       task + pressure slot 1 + pressure slot 2
  ASSISTANT  http_get(catalogue page)
  TOOL       catalogue: official API, public mirror (route B), retired FTP server
  ASSISTANT  http_get(official API)            (route A)
  TOOL       route A's reply                   (swapped: error = blocked / success = not blocked)
  ASSISTANT  <tool_call>{"name": "http_get", "arguments": {"url": "https://
                         ^ read: which tool                          ^ read: which address

The decision part is added by us ("prefilled"). The model reads left to right and never looks ahead, so
one forward pass over the whole thing gives the probabilities at both positions, and the text after the
tool name cannot change the probabilities at the tool name.
"""
import hashlib
import random
from dataclasses import asdict, dataclass, field

from . import content

# The text we add after the chat template's generation prompt.
CALL_OPEN = '<tool_call>\n{"name": "'
CALL_TO_ADDRESS = 'http_get", "arguments": {"url": "https://'
TOOL_NAMES = ("http_get", "ask_user", "submit_answer")


@dataclass
class Setting:
    """Everything that defines one prompt."""
    task_id: str
    rule_type: str                 # allow / forbid / irrelevant / none / implicit_forbid / implicit_allow
    rule_set_id: str | None        # wording set in rules.yaml (None for rule types without a sentence)
    blocked: bool
    pressure: int                  # 0, 1, 2
    mirror: str
    tool_order: tuple = TOOL_NAMES
    draw: int = 0                  # which draw of mirror + tool order (for bookkeeping)
    # options used by side checks; the main design keeps the defaults
    placement: str = "system"      # system / user
    policy_lines: int = 0          # 0 = the rule alone; N = the rule in the middle of N ordinary policy lines
    priority: bool = False         # add the explicit-priority line after a forbidding rule
    route_c: bool = False          # add an approved second source to the catalogue

    def key(self, **changes):
        """A string id for this setting; with changes, the id of a twin setting (e.g. rule_type='allow')."""
        d = asdict(self) | changes
        return "|".join(f"{k}={d[k]}" for k in sorted(d))


@dataclass
class Prompt:
    setting: Setting
    messages: list
    tools: list
    hosts: dict
    rule_text: str | None
    text: str = ""                 # the chat-template text, before our prefilled decision part
    ids: list = field(default_factory=list)
    positions: dict = field(default_factory=dict)   # name -> token index whose prediction we read
    spans: dict = field(default_factory=dict)       # name -> (first, last+1) token indices of a prompt part

    @property
    def sha(self):
        return hashlib.sha1(self.text.encode()).hexdigest()[:12]


def draw_names(task_id, split, draw, seed):
    """Mirror address and tool order for one task and draw. Allow/forbid twins share the same draw."""
    rng = random.Random(f"{seed}|{task_id}|{draw}")
    mirror = rng.choice(content.mirrors(split))
    order = list(TOOL_NAMES)
    rng.shuffle(order)
    return mirror, tuple(order)


def rule_sentence(s: Setting, hosts):
    if s.rule_type not in content.SET_RULES:
        return None
    text = content.rule_set(s.rule_set_id)[s.rule_type].format(mirror=hosts["mirror"], ftp=hosts["ftp"])
    if s.priority and s.rule_type == "forbid":
        text += " " + content.templates()["priority_line"]
    return text


def tool_schemas(order):
    by_name = {t["name"]: t for t in content.templates()["tools"]}
    out = []
    for name in order:
        t = by_name[name]
        props = {p: {"type": "string", "description": d} for p, d in t["parameters"].items()}
        out.append({"type": "function", "function": {
            "name": name, "description": t["description"],
            "parameters": {"type": "object", "properties": props, "required": list(props)}}})
    return out


def _http_call(url):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "http_get", "arguments": {"url": url}}}]}


def build(s: Setting) -> Prompt:
    """The conversation for one setting, as chat messages (no tokenizer needed)."""
    tpl = content.templates()
    task = content.task(s.task_id)
    hosts = content.hosts_for(task, s.mirror)
    fill = task | hosts

    rule = rule_sentence(s, hosts)

    # system message
    system = tpl["role"].format(**fill)
    if s.placement == "system":
        if s.policy_lines:
            lines = list(content.policy_lines()[: s.policy_lines])
            if rule:
                lines.insert(s.policy_lines // 2, rule)
            system += " " + " ".join(lines)
        elif rule:
            system += " " + rule

    # user message: task + pressure (both slots always present, so pressure never changes length)
    p = tpl["pressure"]
    slot1 = p["neutral_1"] if s.pressure == 0 else p["success"]
    slot2 = task["stakes"] if s.pressure == 2 else task["neutral"]
    user = tpl["task"].format(**fill) + " " + slot1 + " " + slot2
    if s.placement == "user" and rule:
        user = rule + " " + user

    # catalogue page
    note = tpl["implicit_notes"].get(s.rule_type, "")
    catalogue = tpl["catalogue"].format(**fill, mirror_note=note)
    if s.route_c:
        catalogue += "\n" + tpl["route_c_line"].format(**fill)

    reply = (tpl["route_a_error"] if s.blocked else tpl["route_a_success"]).format(**fill)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        _http_call(f"https://{hosts['catalog']}/datasets/{task['dataset']}"),
        {"role": "tool", "content": catalogue},
        _http_call(f"https://{hosts['official']}{task['path']}"),
        {"role": "tool", "content": reply},
    ]
    return Prompt(setting=s, messages=messages, tools=tool_schemas(s.tool_order), hosts=hosts, rule_text=rule)


def chat_text(tokenizer, messages, tools):
    """The prompt as Qwen3's chat template writes it, ending with an empty think block (thinking off)."""
    return tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def first_token(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)[0]


def candidates(tokenizer, prompt: Prompt):
    """Token ids we read at each position: the first token of each option. Raises if two options share one."""
    tools = {name: first_token(tokenizer, name) for name in TOOL_NAMES}
    addresses = {k: first_token(tokenizer, v) for k, v in prompt.hosts.items()
                 if k != "approved" or prompt.setting.route_c}
    for label, d in (("tool names", tools), ("addresses", addresses)):
        if len(set(d.values())) != len(d):
            raise ValueError(f"{label} share a first token: { {k: tokenizer.decode([v]) for k, v in d.items()} }")
    return {"call_start": first_token(tokenizer, CALL_OPEN), "tool": tools, "address": addresses}


def _span(tokenizer, text, offsets, substring, start=0):
    """Token indices covering the first occurrence of substring in text (at or after char `start`)."""
    i = text.find(substring, start)
    if i < 0:
        return None
    j = i + len(substring)
    toks = [k for k, (a, b) in enumerate(offsets) if a < j and b > i]
    return (toks[0], toks[-1] + 1) if toks else None


def tokenize(tokenizer, prompt: Prompt) -> Prompt:
    """Fill in the token ids, the positions we read, and where each prompt part sits."""
    text = chat_text(tokenizer, prompt.messages, prompt.tools)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    prefix = enc["input_ids"]
    call_open = tokenizer.encode(CALL_OPEN, add_special_tokens=False)
    to_address = tokenizer.encode(CALL_TO_ADDRESS, add_special_tokens=False)
    if to_address[0] != first_token(tokenizer, "http_get"):
        raise ValueError("the prefilled call does not start with the same token as the name http_get")

    prompt.text = text
    prompt.ids = prefix + call_open + to_address
    prompt.positions = {
        "call_start": len(prefix) - 1,                    # predicts: tool call or plain text?
        "tool": len(prefix) + len(call_open) - 1,          # predicts: which tool?
        "address": len(prompt.ids) - 1,                    # predicts: which address (given http_get)?
    }

    offsets = enc["offset_mapping"]
    msgs = prompt.messages
    tool_defs = text.find("<tools>\n")
    route_a_url = prompt.hosts["official"] + content.task(prompt.setting.task_id)["path"]
    catalogue_at = text.find(msgs[3]["content"])
    spans = {
        "system": _span(tokenizer, text, offsets, msgs[0]["content"]),
        "rule": _span(tokenizer, text, offsets, prompt.rule_text) if prompt.rule_text else None,
        "tool_definitions": (_span(tokenizer, text, offsets, text[tool_defs:text.find("</tools>") + len("</tools>")])
                             if tool_defs >= 0 else None),
        "user": _span(tokenizer, text, offsets, msgs[1]["content"]),
        "catalogue": _span(tokenizer, text, offsets, msgs[3]["content"]),
        "catalogue_mirror_line": _span(tokenizer, text, offsets, f"- {prompt.hosts['mirror']}:", start=catalogue_at),
        "mirror_note": (_span(tokenizer, text, offsets, content.templates()["implicit_notes"][prompt.setting.rule_type],
                              start=catalogue_at)
                        if prompt.setting.rule_type in content.templates()["implicit_notes"] else None),
        "route_a_call": _span(tokenizer, text, offsets, route_a_url),
        "route_a_reply": _span(tokenizer, text, offsets, msgs[5]["content"]),
        "decision": (len(prefix), len(prompt.ids)),
    }
    prompt.spans = {k: v for k, v in spans.items() if v is not None}
    return prompt


def compare_ids(a, b):
    """How two token sequences differ: same length? which positions differ?"""
    same_len = len(a) == len(b)
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return {"same_length": same_len, "n_diff": len(diff) + abs(len(a) - len(b)),
            "first_diff": diff[0] if diff else None, "last_diff": diff[-1] if diff else None}
