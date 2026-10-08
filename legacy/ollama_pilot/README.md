# Ollama pilot (kept for the record)

The first pilot, before the move to Hugging Face models. It sampled 20 replies per setting from
compressed Ollama models, with a dedicated `mirror_fetch` tool and two exit tools (`ask_user`, `report_blocked`).

Replaced in `s0_v0` because:
- `mirror_fetch` handed the forbidden route to the model as a ready-made tool; in the incidents agents found
  routes in their environment with general tools. Now route B is an address in a catalogue, reached with `http_get`.
- Sampling 20 replies hides how strong a choice is (almost every cell was 20/20). Now one forward pass gives exact probabilities.
- Ollama runs compressed models on different software; now the model runs in its original bf16 format with `transformers`.
- Two similar exits split the probability between them; now there is one (`ask_user`).

Results from these runs are in the lab log. To run it again (needs Ollama and `pip install requests`):

```
python legacy/ollama_pilot/pilot.py legacy/ollama_pilot/pilot_14b.yaml --samples 2
```
