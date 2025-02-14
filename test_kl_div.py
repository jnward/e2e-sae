# %%
import torch
import datasets
from datasets import Dataset, IterableDataset
from transformer_lens import HookedTransformer
from gpt2_utils import get_gpt, partial_forward, logits_from_layer
from sae_kit.sae_kit.sparse_autoencoder import SparseAutoencoder

# %%
device = "mps"

# %%
my_dataset = datasets.load_dataset(
    "openwebtext",
    split="train",
    streaming=True,
)
assert isinstance(my_dataset, (Dataset, IterableDataset))

torch.manual_seed(42)
my_dataset.shuffle(seed=42)

# %%
my_gpt = get_gpt(size="small", device=device)

# %%
from transformers import get_constant_schedule_with_warmup
from sae_kit.sae_kit.latent_tracker import LatentTracker
import wandb

def get_data_gen(batch_size: int, dataset: IterableDataset, gpt: HookedTransformer, ctx_len: int = 128):
    data_iter = iter(dataset)
    batch = []
    for example in data_iter:
        tokens = gpt.to_tokens(example["text"])
        tokens = tokens[0, :ctx_len]
        batch.append(tokens)
        if len(batch) == batch_size:
            yield torch.stack(batch)
            batch = []

def get_sae(n_features):
    return SparseAutoencoder(
        d_in=768,
        n_features = n_features,
        device=device,
        k=k,
        n_hidden=0,
    )

# %%
# training loop
import torch.nn.functional as F
from sae_kit.sae_kit.utils import compute_metrics
from tqdm import tqdm

num_features=24576
k=67

# test_sae = get_sae(num_features)
# state_dict = torch.load("models/weights_target-sae.pt", map_location=device)
# test_sae.load_state_dict(state_dict)
test_sae = SparseAutoencoder.from_pretrained(
    "gpt2-small-res-jb",
    "blocks.8.hook_resid_pre",
    1000,
    num_features,
    device=device,
)
# test_sae = get_sae(num_features)

num_steps = 100
my_data_gen = get_data_gen(4, my_dataset, my_gpt)
losses = []
fvus = []

# %%
pbar = tqdm(range(num_steps))
for optimizer_step in pbar:
    batch = next(my_data_gen)
    acts = partial_forward(batch, my_gpt, to_layer_pre=8)

    # _, cache = my_gpt.run_with_cache(batch)
    # test_acts = cache["blocks.8.hook_resid_pre"]
    # assert torch.allclose(acts, test_acts)

    target_logits = logits_from_layer(acts, my_gpt, from_layer_pre=8)

    # test_logits = my_gpt(batch)
    # assert torch.allclose(target_logits, test_logits)

    target_probs = F.softmax(target_logits, dim=-1)

    features = test_sae.encoder(acts)
    reconstructions = test_sae.decoder(features)
    pred_logits = logits_from_layer(reconstructions, my_gpt, from_layer_pre=8)
    pred_probs = F.softmax(pred_logits, dim=-1)
    pred_probs = pred_probs + 1e-10  # Add small epsilon
    
    loss = F.kl_div(torch.log(pred_probs), target_probs, reduction='batchmean')
    losses.append(loss.item())

    e = reconstructions - acts
    total_variance = (acts - acts.mean(0)).pow(2).sum()
    squared_error = e.pow(2)
    fvu = squared_error.sum() / total_variance
    fvus.append(fvu.item())

    avg_loss = sum(losses) / len(losses)
    avg_fvu = sum(fvus) / len(fvus)

    l0 = (features > 0).float().sum(-1)[0]

    pbar.set_description(f"Loss: {avg_loss:.4f}, FVU: {avg_fvu:.4f}")


avg_loss = sum(losses) / len(losses)
print(f"Average loss: {avg_loss}")
# %%