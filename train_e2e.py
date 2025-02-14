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
def tokenize_example(example: str, gpt: HookedTransformer):
    tokens = gpt.to_tokens(example)
    print(tokens)

tokenize_example("Hello, world!", my_gpt)
# %%
from torch.optim import Adam
from transformers import get_constant_schedule_with_warmup
from sae_kit.sae_kit.latent_tracker import LatentTracker
import wandb

def get_data_gen(batch_size: int, dataset: IterableDataset, gpt: HookedTransformer, ctx_len: int = 1024):
    data_iter = iter(dataset)
    batch = []
    for example in data_iter:
        tokens = gpt.to_tokens(example["text"])
        tokens = tokens[0, :ctx_len]
        if tokens.shape[0] < ctx_len:
            continue
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

def get_optimizer(sae, lr=1e-4):
    return Adam(sae.parameters(), lr=lr)

def get_scheduler(optimizer, num_warmup_steps=100):
    return get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
    )

def get_run(name):
    return wandb.init(
        project="gpt2-e2e-sae-multi-train",
        name=name,
    )

def get_latent_tracker(sae):
    return LatentTracker(
        sae.n_features,
        device=device,
        dead_threshold=10_000_000
    )

# %%
# training loop
import torch.nn.functional as F
from sae_kit.sae_kit.utils import compute_metrics
from tqdm import tqdm

features_to_train = [768]
k=67
num_optimizer_steps = 24_000 * 2
virtual_batch_size = 16
actual_batch_size = 1
accumulation_steps = virtual_batch_size // actual_batch_size
total_forward_passes = num_optimizer_steps * accumulation_steps
alpha = 1/32

sae_paths = []
saes = []
optimizers = []
schedulers = []
# runs = []
data = []
latent_trackers = []

run = get_run(f"{features_to_train[-1]}-{k}-e2e")
my_data_gen = get_data_gen(actual_batch_size, my_dataset, my_gpt)


for n_features in features_to_train:
    sae_name_ = f"{n_features}-{k}-e2e"
    sae_path_ = f"trained_saes/{sae_name_}.pt"
    sae_ = get_sae(n_features)
    optimizer_ = get_optimizer(sae_)
    scheduler_ = get_scheduler(optimizer_)
    latent_tracker_ = get_latent_tracker(sae_)

    sae_paths.append(sae_path_)
    saes.append(sae_)
    optimizers.append(optimizer_)
    schedulers.append(scheduler_)
    latent_trackers.append(latent_tracker_)
    data.append([])

# %%
pbar = tqdm(range(num_optimizer_steps))
for optimizer_step in pbar:
    for optimizer in optimizers:
        optimizer.zero_grad()
    for acc_step in range(accumulation_steps):
        batch = next(my_data_gen)
        with torch.no_grad():
            acts = partial_forward(batch, my_gpt, to_layer_pre=8)
            target_logits = logits_from_layer(acts, my_gpt, from_layer_pre=8)
            target_probs = F.softmax(target_logits, dim=-1)
        for i, (sae_path, sae, run_dicts, latent_tracker) in enumerate(zip(sae_paths, saes, data, latent_trackers)):
            # Accumulate gradients over multiple forward passes
            with torch.set_grad_enabled(True):
                features = sae.encode(acts)
                reconstruction = sae.decode(features)
                pred_logits = logits_from_layer(reconstruction, my_gpt, from_layer_pre=8)
                pred_log_probs = F.log_softmax(pred_logits, dim=-1)
                
                main_loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean") / accumulation_steps

                latent_tracker.update(features.reshape(-1, sae.n_features))
                dead_latents = latent_tracker.get_dead_latents()

                # aux_loss = auxiliary_loss(dead_latents, error, sae) / accumulation_steps
                aux_loss = torch.zeros_like(main_loss)
                
                # Combined loss
                loss = main_loss + alpha * aux_loss
            
                if torch.isnan(aux_loss):
                    loss = main_loss  # Zero out aux loss if NaN

                loss.backward()

            if acc_step == 0 and optimizer_step % 10 == 0:
                l0, fvu = compute_metrics(acts, features, reconstruction)
                explained_variance = 1 - fvu

                run_data = {
                        "main_loss": main_loss.item() * accumulation_steps,
                        "loss": loss.item() * accumulation_steps,
                        "aux_loss": aux_loss.item() * accumulation_steps,
                        "l0": l0,
                        "variance_explained": explained_variance,
                        # "lr": scheduler.get_last_lr()[0],
                        "dead_latents": dead_latents.sum().item(),
                        "step": optimizer_step,
                    }
                run_dicts.append(run_data)
                if i == len(features_to_train) - 1:
                    pbar.set_description(
                        f"Loss: {main_loss.item() * accumulation_steps:.4f}, "
                        f"ve: {explained_variance:.4f}"
                    )
                    del(run_data["step"])
                    run.log(run_data, step=optimizer_step)


    for sae_path, sae, optimizer, scheduler in zip(sae_paths, saes, optimizers, schedulers):
        # Update weights after accumulating all gradients
        sae.decoder.normalize_weights()
        sae.decoder.remove_parallel_gradient_component()
        optimizer.step()
        scheduler.step()

        if optimizer_step and optimizer_step % 3000 == 0 or optimizer_step == num_optimizer_steps:
            print(f"Saving SAE to {sae_path}")
            torch.save(sae.state_dict(), sae_path)

run.finish()

for n_features, data_dicts in list(zip(features_to_train, data))[:-1]:
    sae_name = f"{n_features}-{k}-e2e"
    run = get_run(sae_name)
    for data_dict in data_dicts:
        step = data_dict["step"]
        del(data_dict["step"])
        run.log(data_dict, step=step)
    run.finish()

# %%
