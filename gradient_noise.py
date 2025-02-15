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

with torch.no_grad():
    acts = partial_forward(tokens, my_gpt, layer)
    true_logits = logits_from_layer(acts, my_gpt, layer)
    true_logprobs = F.log_softmax(true_logits, dim=-1)
print(true_logits.shape, acts.shape)

noise_std = 0.0001
# noise = torch.randn_like(acts) * noise_std
# test_acts = noise
# test_acts = acts.clone() + torch.randn_like(acts) * noise_std # need to add noise to get gradients
test_acts = acts.clone()
test_acts.requires_grad = True

ce_baseline = (true_logprobs.exp() * true_logprobs).sum(dim=-1).mean()

num_steps = 10000
optimizer = torch.optim.Adam([test_acts], lr=5e-3)

pbar = tqdm(range(num_steps))
for step in pbar:
    optimizer.zero_grad()
    
    # with torch.set_grad_enabled(True):
    pred_logits = logits_from_layer(test_acts, my_gpt, layer)
    pred_logprobs = F.log_softmax(pred_logits, dim=-1)

    recon_mse = F.mse_loss(test_acts, acts)  # want to maximize
    logit_mse = F.mse_loss(pred_logits, true_logits)  # want to minimize
    kl = F.kl_div(pred_logprobs, true_logprobs, log_target=True, reduction="none")  # want to minimize
    kl = kl.sum(dim=-1).mean()
    ce_diff = (true_logprobs.exp() * pred_logprobs).sum(dim=-1).mean() - ce_baseline
    tv_loss = torch.abs(test_acts[:, 1:, :] - test_acts[:, :-1, :]).mean()

    late_acts = acts[0, 20:]
    late_test_acts = test_acts[0, 20:]
    numerator = ((late_acts - late_test_acts) ** 2).sum()
    denom = ((late_acts - late_acts.mean()) ** 2).sum()
    fvu = numerator / denom
    # loss = 10 * logit_mse - recon_mse
    loss = kl + tv_loss
    # loss = -recon_mse
    # loss = 10 * kl - recon_mse
    if kl.item() > 0.1:
        break
    loss.backward()
    optimizer.step()



    # pbar.set_description(f"Loss: {loss.item():.4f} Logit MSE: {logit_mse.item():.4f} Recon MSE: {recon_mse.item():.4f}")
    # pbar.set_description(f"Loss: {loss.item():.4f} Logit MSE: {logit_mse.item():.4f} Recon MSE: {recon_mse.item():.4f} FVU: {fvu.item():.4f}")
    # pbar.set_description(f"Loss: {loss.item():.4f} KL: {kl.item():.4f} Recon MSE: {recon_mse.item():.4f} FVU: {fvu.item():.4f}")
    # pbar.set_description(f"Loss: {loss.item():.4f} CE_diff: {ce_diff.item():.4f} KL: {kl.item():.4f} Recon MSE: {recon_mse.item():.4f} FVU: {fvu.item():.4f}")
    pbar.set_description(f"Loss: {loss.item():.4f} TV: {tv_loss.item():.4f} KL: {kl.item():.4f} Recon MSE: {recon_mse.item():.4f} FVU: {fvu.item():.4f}")

# %%