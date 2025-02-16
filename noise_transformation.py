# %%
import torch
import torch.nn.functional as F
import datasets
from tqdm import tqdm
from gpt2_utils import partial_forward, logits_from_layer, get_gpt

device="cuda"
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPTransformation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_branch = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False)
        )
        self.skip_branch = nn.Linear(dim, dim, bias=False)
        # self._init_weights(dim)

    def _init_weights(self, dim):
        with torch.no_grad():
            # Initialize skip branch to identity
            self.skip_branch.weight.copy_(torch.eye(dim, device=self.skip_branch.weight.device, dtype=self.skip_branch.weight.dtype))
            # Initialize MLP branch weights to small values
            for layer in self.mlp_branch:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.mul_(0.01)

    def forward(self, x):
        # return self.skip_branch(x) + self.mlp_branch(x)
        return x + self.mlp_branch(x)

import torch
import torch.nn as nn

class ConstantBias(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(dim) * 0.1)
    
    def forward(self, x):
        # x is assumed to have shape [*, dim] (e.g. [batch, context, dim])
        return x + self.bias


train_linear = False
if train_linear:
    transformation = torch.nn.Linear(768, 768, bias=False, device=device, dtype=torch.float32)
    # initialize to identity
    transformation.weight.data *= 0.01
    transformation.weight.data += torch.eye(768, device=device, dtype=torch.float32)
else:
    transformation = MLPTransformation(768)
    # transformation = ConstantBias(768)
    transformation.to(device)

num_steps = 10000
optimizer = torch.optim.Adam(transformation.parameters(), lr=1e-4)


batch_size = 64

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
    # if step < 0:
    #     loss = 10 * kl + recon_mse
    # else:
    loss = 4 * kl - fvu
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
U, S, V = torch.svd(transformation.weight.data)

# plot cumulative variance explained
cum_var = (S**2).cumsum(dim=0) / (S**2).sum(dim=0)
import plotly.express as px

px.line(y=cum_var.cpu().numpy(), title="Cumulative Variance Explained")
# %%
px.line((S**2).cpu().numpy(), title="Singular Values")
# %%

fvus = []
kls = []
residuals = []
pbar = tqdm(range(16))

transformation.to(device)
for step in pbar:
    with torch.no_grad():
        acts, true_logits, true_logprobs = get_batch(batch_size)

        pred_acts = transformation(acts)
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

        fvus.append(fvu.item())
        kls.append(kl.item())

        residual = acts - pred_acts
        residuals.append(residual[0, 20:])

residuals = torch.stack(residuals)
print(residuals.shape)
print("FVU: ", torch.tensor(fvus).mean().item())
print("KL: ", torch.tensor(kls).mean().item())


# %%
import plotly.express as px
U, S, V = torch.svd(residuals.reshape(-1, 768))

cum_var = (S**2).cumsum(dim=0) / (S**2).sum(dim=0)
px.line(y=cum_var.cpu().numpy(), title="Cumulative Variance Explained")

# %%
dominant_direction = V[:, 0]  # shape [768]

# Optionally, compute the variance explained by this direction:
var_explained = (S[0]**2) / (S**2).sum()
print("Variance explained by the dominant direction:", var_explained.item())
# %%
(dominant_direction @ my_gpt.W_U).abs().sum()
# %%
random_direction = torch.randn_like(dominant_direction)
my_gpt.unembed(my_gpt.ln_final(random_direction)).norm()
# return gpt.unembed(x)
# %%
from sae_lens import SAE

my_sae, _, _ = SAE.from_pretrained(
    "gpt2-small-res-jb",
    "blocks.8.hook_resid_pre",
    device=device
)
# %%
# compute cosine sim
random_direction = torch.randn_like(dominant_direction)
# direction_of_interest = random_direction
direction_of_interest = dominant_direction

product = direction_of_interest @ my_sae.W_dec.T
cosine_sim = product / (direction_of_interest.norm() * my_sae.W_dec.norm(dim=1))
cosine_sim.max()
# %%
