# %%
import torch
from transformer_lens import HookedTransformer


def get_gpt(size: str = "small", device: str = "cuda") -> HookedTransformer:
    return HookedTransformer.from_pretrained(
        f"gpt2-{size}",
        device=device,
        dtype=torch.bfloat16,
    )

def partial_forward(
    tokens: torch.Tensor,
    gpt: HookedTransformer,
    to_layer_pre: int,
):
    # x = gpt.embed(tokens) + gpt.pos_embed[: tokens.size(1)]
    x, _, _, _ = gpt.input_to_embed(tokens)
    for i in range(to_layer_pre):
        x = gpt.blocks[i](x)
    return x

def logits_from_layer(
    acts: torch.Tensor,
    gpt: HookedTransformer,
    from_layer_pre: int,
):
    x = acts
    for i in range(from_layer_pre, gpt.cfg.n_layers):
        x = gpt.blocks[i](x)
    x = gpt.ln_final(x)
    return gpt.unembed(x)
