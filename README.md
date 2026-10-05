# Representation Diagnostics: Detour - experiments

How a stated rule acts on a blocked agent's choice. 

## Layout

```
detour/
  README.md      this file
  LOG.md         lab log & findings
  lab.py         makes a new run folder for every run
  pilot.py       Ollama pilot
  configs/       one file per experiment; all settings live here
  runs/          created automatically; one folder per run; never edited by hand
  data/          frozen inputs (prompt sets later); a file here is never changed, only replaced by a new name
  notebooks/     looking at results only
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install requests pyyaml
ollama pull <model>
```

## Run

```
python pilot.py configs/pilot_14b.yaml --samples 2    # dry run: checks everything works
python pilot.py configs/pilot_14b.yaml                # full run: 96 settings x 20 samples
```

---


*Research by Anastasia Chernysheva*