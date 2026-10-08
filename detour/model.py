"""Load Qwen3 in its original number format (bf16) for exact next-token probabilities.

On a Mac the weights go straight to the GPU ("mps") so they are never held twice in memory:
Qwen3-8B in bf16 needs about 16 GB, which fits in 32 GB of unified memory only if loaded once.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def pick_device(name="auto"):
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def check_dtype(device, dtype):
    """Fail early with a clear message if this device can't compute in this format."""
    try:
        x = torch.ones(4, 4, dtype=dtype, device=device)
        (x @ x).sum().item()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"{device} cannot compute in {dtype}: {e}\n"
            "On a Mac, bf16 needs macOS 14 or newer and a recent PyTorch (pip install -U torch). "
            "Do not fall back to float32 for Qwen3-8B on a 32 GB Mac: it needs about 32 GB.") from e


def load_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)


def load(model_name, dtype="bfloat16", device="auto", tokenizer=None):
    device = pick_device(device)
    torch_dtype = DTYPES[dtype]
    check_dtype(device, torch_dtype)
    tok = load_tokenizer(tokenizer or model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device)
    model.eval()
    return model, tok, device


def tiny_random(tokenizer, device="cpu"):
    """A 2-layer Qwen3 with random weights and the real vocabulary. For testing the code only:
    its probabilities mean nothing."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
    return Qwen3ForCausalLM(cfg).to(device).eval()


def describe(model, device):
    c = model.config
    return {"model": getattr(c, "_name_or_path", ""), "layers": c.num_hidden_layers,
            "heads": c.num_attention_heads, "hidden": c.hidden_size,
            "dtype": str(next(model.parameters()).dtype), "device": device,
            "torch": torch.__version__}
