# %%
import torch
import torch.nn.functional as F
import datasets
from tqdm import tqdm
from gpt2_utils import partial_forward, logits_from_layer, get_gpt

device="mps"
layer = 8

my_gpt = get_gpt(size="small", device=device, dtype=torch.float32)
my_dataset = datasets.load_dataset(
    "openwebtext",
    split="train",
    streaming=True,
)
my_data_iter = iter(my_dataset)

# %%
example = next(my_data_iter)
tokens = my_gpt.to_tokens(example["text"])
tokens = tokens[0, :128]

def get_batch(batch_size):
    all_acts = []
    all_true_logits = []
    all_true_logprobs = []
    for i in range(batch_size):
        example = next(my_data_iter)
        tokens = my_gpt.to_tokens(example["text"])
        tokens = tokens[0, :128]
        with torch.no_grad():
            acts = partial_forward(tokens, my_gpt, layer)
            true_logits = logits_from_layer(acts, my_gpt, layer)
            true_logprobs = F.log_softmax(true_logits, dim=-1)
        all_acts.append(acts.squeeze(0))
        all_true_logits.append(true_logits.squeeze(0))
        all_true_logprobs.append(true_logprobs.squeeze(0))
    all_acts = torch.stack(all_acts)
    all_true_logits = torch.stack(all_true_logits)
    all_true_logprobs = torch.stack(all_true_logprobs)
    
    return all_acts, all_true_logits, all_true_logprobs

# noise_std = 0.0001
# noise = torch.randn_like(acts) * noise_std
# test_acts = noise
# test_acts = acts.clone() + torch.randn_like(acts) * noise_std # need to add noise to get gradients
# test_acts = acts.clone()
# test_acts.requires_grad = True

# ce_baseline = (true_logprobs.exp() * true_logprobs).sum(dim=-1).mean()

transformation = torch.nn.Linear(768, 768, device=device, dtype=torch.float32)

num_steps = 10000
optimizer = torch.optim.Adam(transformation.parameters(), lr=1e-4)
# initialize to identity
transformation.weight.data = torch.eye(768, device=device, dtype=torch.float32)

batch_size = 8

pbar = tqdm(range(num_steps))
for step in pbar:
    optimizer.zero_grad()

    acts, true_logits, true_logprobs = get_batch(batch_size)

    pred_acts = transformation(acts)
    
    # with torch.set_grad_enabled(True):
    pred_logits = logits_from_layer(pred_acts, my_gpt, layer)
    pred_logprobs = F.log_softmax(pred_logits, dim=-1)

    recon_mse = F.mse_loss(pred_acts, acts)  # want to maximize
    logit_mse = F.mse_loss(pred_logits, true_logits)  # want to minimize
    kl = F.kl_div(pred_logprobs, true_logprobs, log_target=True, reduction="none")  # want to minimize
    kl = kl.sum(dim=-1).mean()
    # ce_diff = (true_logprobs.exp() * pred_logprobs).sum(dim=-1).mean() - ce_baseline
    # tv_loss = torch.abs(test_acts[:, 1:, :] - test_acts[:, :-1, :]).mean()

    late_acts = acts[0, 20:]
    late_pred_acts = pred_acts[0, 20:]
    numerator = ((late_acts - late_pred_acts) ** 2).sum()
    denom = ((late_acts - late_acts.mean()) ** 2).sum()
    fvu = numerator / denom
    # loss = 10 * logit_mse - recon_mse
    # loss = kl + tv_loss
    # loss = -recon_mse
    if step < 0:
        loss = 10 * kl + recon_mse
    else:
        loss = 10 * kl - recon_mse
    # if kl.item() > 0.1:
    #     break
    loss.backward()
    optimizer.step()



    # pbar.set_description(f"Loss: {loss.item():.4f} Logit MSE: {logit_mse.item():.4f} Recon MSE: {recon_mse.item():.4f}")
    # pbar.set_description(f"Loss: {loss.item():.4f} Logit MSE: {logit_mse.item():.4f} Recon MSE: {recon_mse.item():.4f} FVU: {fvu.item():.4f}")
    pbar.set_description(f"Loss: {loss.item():.4f} KL: {kl.item():.4f} Recon MSE: {recon_mse.item():.4f} FVU: {fvu.item():.4f}")
    # pbar.set_description(f"Loss: {loss.item():.4f} CE_diff: {ce_diff.item():.4f} KL: {kl.item():.4f} Recon MSE: {recon_mse.item():.4f} FVU: {fvu.item():.4f}")
    # pbar.set_description(f"Loss: {loss.item():.4f} TV: {tv_loss.item():.4f} KL: {kl.item():.4f} Recon MSE: {recon_mse.item():.4f} FVU: {fvu.item():.4f}")

# %%