"""Code for the Detour study: how a stated rule acts on a blocked agent's choice.

Modules
  config   load a run config (with an optional `base:` config it extends)
  content  load the hand-written prompt material in content/
  prompts  build one prompt from its parts; find the token positions we read
  grid     list every prompt a config asks for
  model    load Qwen3 in its original format (bf16) on the Mac GPU, an NVIDIA GPU or the CPU
  score    one forward pass -> the probabilities at the decision point
  checks   the separate question checks ("forks")
"""
