#!/usr/bin/env python3
"""
Export all figures from the tutorial for the LaTeX companion document.

Usage:
    python export_figures.py

Generates PDF figures in figures/ that correspond to the \includegraphics
calls in tutorial_notes.tex. Section numbering matches the tutorial.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# Shared plotting style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#333333",
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 150,
})

# ==================================================================
# Setup: inference problem (shared across sections)
# ==================================================================
TRUE_A, TRUE_PHI = 2.2, 1.8
SIGMA = 0.8
N_DATA = 8

rng_data = np.random.default_rng(77)
t_obs = np.array([(i + 0.5) / N_DATA for i in range(N_DATA)])
data_obs = TRUE_A * np.sin(2 * np.pi * t_obs + TRUE_PHI) + SIGMA * rng_data.normal(size=N_DATA)

A_RANGE = (0.2, 4.5)
PHI_RANGE = (0.0, 2 * np.pi)
GRID_N = 100

def log_posterior(A, phi):
    if A < A_RANGE[0] or A > A_RANGE[1] or phi < PHI_RANGE[0] or phi > PHI_RANGE[1]:
        return -np.inf
    model = A * np.sin(2 * np.pi * t_obs + phi)
    return -np.sum((data_obs - model) ** 2) / (2 * SIGMA ** 2)

# Exact posterior grid
A_vals = np.linspace(A_RANGE[0], A_RANGE[1], GRID_N)
phi_vals = np.linspace(PHI_RANGE[0], PHI_RANGE[1], GRID_N)
A_grid, phi_grid = np.meshgrid(A_vals, phi_vals)
lp_grid = np.array([[log_posterior(A_grid[j, i], phi_grid[j, i])
                      for i in range(GRID_N)] for j in range(GRID_N)])
lp_grid[~np.isfinite(lp_grid)] = -1e10
prob_grid = np.exp(lp_grid - np.max(lp_grid))

# Marginals
dA = (A_RANGE[1] - A_RANGE[0]) / GRID_N
dP = (PHI_RANGE[1] - PHI_RANGE[0]) / GRID_N
exact_margA = np.sum(prob_grid, axis=0)
exact_margA /= np.sum(exact_margA) * dA
exact_margPhi = np.sum(prob_grid, axis=1)
exact_margPhi /= np.sum(exact_margPhi) * dP

# ==================================================================
# Flow implementations (copied from tutorial.py)
# ==================================================================
def init_planar(n_layers, seed=55):
    rng = np.random.default_rng(seed)
    return [{"u": rng.normal(0, 0.3, size=2), "w": rng.normal(0, 0.3, size=2),
             "b": rng.normal(0, 0.1)} for _ in range(n_layers)]

def planar_forward(z, params):
    x = z.copy(); log_det = 0.0
    for layer in params:
        u, w, b = layer["u"], layer["w"], layer["b"]
        dot = x @ w + b; th = np.tanh(dot); dth = 1 - th ** 2
        psi = dth[:, None] * w[None, :]
        log_det = log_det + np.log(np.abs(1 + psi @ u) + 1e-10)
        x = x + np.outer(th, u)
    return x, log_det

def init_radial(n_layers, seed=55):
    rng = np.random.default_rng(seed)
    return [{"z0": rng.normal(0, 0.3, size=2), "log_alpha": rng.normal(0, 0.1),
             "beta": rng.normal(0, 0.2)} for _ in range(n_layers)]

def radial_forward(z, params):
    x = z.copy(); log_det = np.zeros(len(z))
    for layer in params:
        z0, log_alpha, beta = layer["z0"], layer["log_alpha"], layer["beta"]
        alpha = np.exp(log_alpha) + 0.01
        diff = x - z0[None, :]; r = np.sqrt(np.sum(diff ** 2, axis=1)) + 1e-8
        h = 1.0 / (alpha + r); hp = -h ** 2
        scale = 1 + beta * h; det = scale * (scale + r * beta * hp)
        log_det += np.log(np.abs(det) + 1e-10)
        x = x + beta * (h[:, None] * diff)
    return x, log_det

def init_coupling(n_layers, seed=55):
    rng = np.random.default_rng(seed)
    return [{"dim": i % 2, "sa1": rng.normal(0, 0.3), "sa2": 0.5 + rng.normal(0, 0.2),
             "sa3": rng.normal(0, 0.1), "sa4": rng.normal(0, 0.1),
             "tb1": rng.normal(0, 0.3), "tb2": 0.5 + rng.normal(0, 0.2),
             "tb3": rng.normal(0, 0.1), "tb4": rng.normal(0, 0.1)}
            for i in range(n_layers)]

def coupling_forward(z, params):
    x = z.copy(); log_det = np.zeros(len(z))
    for L in params:
        d = L["dim"]; other = 1 - d; cond = x[:, d]
        log_s = L["sa1"] * np.tanh(L["sa2"] * cond + L["sa3"]) + L["sa4"]
        t = L["tb1"] * np.tanh(L["tb2"] * cond + L["tb3"]) + L["tb4"]
        x[:, other] = x[:, other] * np.exp(log_s) + t
        log_det += log_s
    return x, log_det

def flow_to_physical(x, ranges=None):
    if ranges is None: ranges = [A_RANGE, PHI_RANGE]
    out = np.empty_like(x)
    for d in range(2):
        sig = 1 / (1 + np.exp(-np.clip(x[:, d], -10, 10)))
        out[:, d] = ranges[d][0] + (ranges[d][1] - ranges[d][0]) * sig
    return out

def log_sigmoid_jac(x, ranges=None):
    if ranges is None: ranges = [A_RANGE, PHI_RANGE]
    total = np.zeros(x.shape[0])
    for d in range(2):
        xc = np.clip(x[:, d], -10, 10); sig = 1 / (1 + np.exp(-xc))
        total += np.log(sig * (1 - sig) + 1e-10) + np.log(ranges[d][1] - ranges[d][0])
    return total

def compute_loss(z_batch, params, forward_fn, log_p_fn, ranges=None):
    x, log_det = forward_fn(z_batch, params)
    phys = flow_to_physical(x, ranges)
    log_sj = log_sigmoid_jac(x, ranges)
    log_p_vals = np.array([log_p_fn(phys[i, 0], phys[i, 1]) for i in range(len(z_batch))])
    return np.mean(-log_det - log_sj - log_p_vals)

def train_flow(init_fn, forward_fn, log_p_fn, n_layers, n_epochs,
               batch_size=100, lr=0.02, lr_decay=0.001, seed=55, ranges=None):
    import copy
    params = init_fn(n_layers, seed=seed)
    rng = np.random.default_rng(999); loss_history = []; eps = 1e-4
    for epoch in range(1, n_epochs + 1):
        current_lr = lr * np.exp(-epoch * lr_decay)
        z_batch = rng.normal(size=(batch_size, 2))
        base_loss = compute_loss(z_batch, params, forward_fn, log_p_fn, ranges)
        loss_history.append(base_loss)
        for layer in params:
            for key in layer:
                val = layer[key]
                if isinstance(val, np.ndarray):
                    for idx in np.ndindex(val.shape):
                        val[idx] += eps
                        lp = compute_loss(z_batch, params, forward_fn, log_p_fn, ranges)
                        val[idx] -= 2 * eps
                        lm = compute_loss(z_batch, params, forward_fn, log_p_fn, ranges)
                        val[idx] += eps
                        val[idx] -= current_lr * np.clip((lp - lm) / (2 * eps), -2, 2)
                elif isinstance(val, (float, np.floating)):
                    layer[key] = val + eps
                    lp = compute_loss(z_batch, params, forward_fn, log_p_fn, ranges)
                    layer[key] = val - eps
                    lm = compute_loss(z_batch, params, forward_fn, log_p_fn, ranges)
                    layer[key] = val
                    layer[key] -= current_lr * np.clip((lp - lm) / (2 * eps), -2, 2)
        if epoch % 100 == 0:
            print(f"  epoch {epoch}/{n_epochs}, loss={base_loss:.2f}")
    return params, loss_history

def run_mcmc(n_steps, prop_scale=0.25, seed=123, start=None):
    rng = np.random.default_rng(seed)
    if start is None: start = np.array([1.0, 4.0])
    chain = np.empty((n_steps + 1, 2)); chain[0] = start
    current_lp = log_posterior(start[0], start[1]); n_accept = 0
    for i in range(n_steps):
        proposal = chain[i] + prop_scale * rng.normal(size=2)
        prop_lp = log_posterior(proposal[0], proposal[1])
        if np.log(rng.uniform()) < prop_lp - current_lp:
            chain[i + 1] = proposal; current_lp = prop_lp; n_accept += 1
        else: chain[i + 1] = chain[i]
    return chain, n_accept / n_steps

def compute_ess(chain, burnin=200):
    vals = chain[burnin:, 0]; n = len(vals)
    if n < 20: return 0
    mean = np.mean(vals); c0 = np.mean((vals - mean) ** 2)
    if c0 == 0: return n
    tau_sum = 0
    for lag in range(1, min(n, 500)):
        ck = np.mean((vals[:-lag] - mean) * (vals[lag:] - mean))
        if ck / c0 < 0.05: break
        tau_sum += ck / c0
    return max(1, int(n / (1 + 2 * tau_sum)))


# ==================================================================
# Figure 1: Data and posterior (Section 1)
# ==================================================================
print("Generating sec1_data_and_posterior.pdf ...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
t_fine = np.linspace(0, 1, 200)
ax.plot(t_fine, TRUE_A * np.sin(2 * np.pi * t_fine + TRUE_PHI), "C0", lw=1.5, label="true signal")
ax.errorbar(t_obs, data_obs, yerr=SIGMA, fmt="o", color="C1", markersize=5, capsize=3, label="data")
ax.set_xlabel("$t$"); ax.set_ylabel("$d(t)$")
ax.legend(fontsize=9); ax.set_title("Observed data")

ax = axes[1]
ax.contourf(A_vals, phi_vals, prob_grid, levels=20, cmap="viridis")
ax.axvline(TRUE_A, color="white", ls="--", lw=0.8, alpha=0.5)
ax.axhline(TRUE_PHI, color="white", ls="--", lw=0.8, alpha=0.5)
ax.plot(TRUE_A, TRUE_PHI, "w+", markersize=10, mew=1.5)
ax.set_xlabel("Amplitude $A$"); ax.set_ylabel(r"Phase $\varphi$")
ax.set_title(r"Posterior $p(A, \varphi \mid d)$")
fig.tight_layout()
fig.savefig(OUT / "sec1_data_and_posterior.pdf", bbox_inches="tight")
plt.close(fig)


# ==================================================================
# Figure 2: Layer deformation (Section 3)
# ==================================================================
print("Generating sec3_layer_deformation.pdf ...")
N_LAYERS_VIZ = 12
params_viz = init_planar(N_LAYERS_VIZ, seed=55)
rng_viz = np.random.default_rng(42)
z_viz = rng_viz.normal(size=(2000, 2))
show_at = [0, 1, 3, 6, 9, 12]

fig, axes = plt.subplots(1, len(show_at), figsize=(12, 2.5))
for idx, k in enumerate(show_at):
    ax = axes[idx]
    if k == 0: pts = z_viz
    else: pts, _ = planar_forward(z_viz, params_viz[:k])
    ax.scatter(pts[:, 0], pts[:, 1], s=0.3, alpha=0.3, c="C2", rasterized=True)
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect("equal")
    ax.set_title(f"{'Base $z$' if k == 0 else f'{k} layers'}", fontsize=9)
    ax.tick_params(labelsize=6)
    if idx > 0: ax.set_yticklabels([])
fig.suptitle(f"Gaussian $\\to$ {N_LAYERS_VIZ} planar layers", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "sec3_layer_deformation.pdf", bbox_inches="tight")
plt.close(fig)


# ==================================================================
# Figure 3: Training (Section 4)
# ==================================================================
print("Generating sec4_training.pdf (this takes a minute) ...")
NL_TRAIN, NE_TRAIN = 8, 800
trained_params, loss_hist = train_flow(init_planar, planar_forward, log_posterior,
                                        n_layers=NL_TRAIN, n_epochs=NE_TRAIN)
rng_s = np.random.default_rng(42)
z_s = rng_s.normal(size=(3000, 2))
fx, _ = planar_forward(z_s, trained_params)
fphys = flow_to_physical(fx)
fmask = ((fphys[:, 0] >= A_RANGE[0]) & (fphys[:, 0] <= A_RANGE[1]) &
         (fphys[:, 1] >= PHI_RANGE[0]) & (fphys[:, 1] <= PHI_RANGE[1]))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
ax = axes[0]
ax.plot(range(1, len(loss_hist) + 1), loss_hist, "C2", lw=0.8)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (KL)"); ax.set_title("Training loss")
if len(loss_hist) > 50:
    ymin, ymax = min(loss_hist[50:]), max(loss_hist[50:])
    ax.set_ylim(ymin - 0.2 * (ymax - ymin), ymax + 0.2 * (ymax - ymin))

ax = axes[1]
ax.contourf(A_vals, phi_vals, prob_grid, levels=20, cmap="viridis", alpha=0.8)
ax.plot(TRUE_A, TRUE_PHI, "w+", markersize=10, mew=1.5)
ax.set_xlabel("$A$"); ax.set_ylabel(r"$\varphi$"); ax.set_title("Exact posterior")

ax = axes[2]
ax.contourf(A_vals, phi_vals, prob_grid, levels=20, cmap="viridis", alpha=0.2)
ax.scatter(fphys[fmask, 0], fphys[fmask, 1], s=0.5, alpha=0.3, c="C2", rasterized=True)
ax.set_xlim(A_RANGE); ax.set_ylim(PHI_RANGE)
ax.set_xlabel("$A$"); ax.set_ylabel(r"$\varphi$")
ax.set_title(f"Flow samples ({NL_TRAIN}L, {NE_TRAIN} epochs)")
ax.plot(TRUE_A, TRUE_PHI, "w+", markersize=10, mew=1.5)
fig.tight_layout()
fig.savefig(OUT / "sec4_training.pdf", bbox_inches="tight")
plt.close(fig)


# ==================================================================
# Figure 4: MCMC vs flow (Section 5)
# ==================================================================
print("Generating sec5_mcmc_vs_flow.pdf ...")
N_MCMC = 5000; BURNIN = 200
chain, acc_rate = run_mcmc(N_MCMC)
ess = compute_ess(chain, BURNIN)
fphys_valid = fphys[fmask]

fig, axes = plt.subplots(3, 2, figsize=(10, 10))
for col, (title, samples, color, stats) in enumerate([
    ("MCMC (Metropolis-Hastings)", chain[BURNIN:], "C1",
     f"steps: {N_MCMC}\nESS: {ess}\naccept: {acc_rate:.0%}"),
    (f"Normalizing Flow ({NL_TRAIN}L)", fphys_valid, "C2",
     f"samples: {len(fphys_valid)}\nESS: {len(fphys_valid)}"),
]):
    ax = axes[0, col]
    ax.contourf(A_vals, phi_vals, prob_grid, levels=15, cmap="viridis", alpha=0.25)
    ax.scatter(samples[:, 0], samples[:, 1], s=0.4, alpha=0.25, color=color, rasterized=True)
    ax.set_xlim(A_RANGE); ax.set_ylim(PHI_RANGE)
    ax.set_xlabel("$A$"); ax.set_ylabel(r"$\varphi$")
    ax.set_title(title, fontsize=10, color=color)
    ax.text(0.97, 0.97, stats, transform=ax.transAxes, fontsize=8,
            va="top", ha="right", family="monospace")
    ax.plot(TRUE_A, TRUE_PHI, "w+", markersize=8, mew=1)

for col, (samples, color, label) in enumerate([
    (chain[BURNIN:, 0], "C1", "MCMC"), (fphys_valid[:, 0], "C2", "Flow")]):
    ax = axes[1, col]
    ax.hist(samples, bins=50, density=True, alpha=0.5, color=color, label=label)
    ax.plot(A_vals, exact_margA, "C0", lw=1.5, label="exact")
    ax.axvline(TRUE_A, color="gray", ls="--", lw=0.7)
    ax.set_xlabel("$A$"); ax.set_ylabel("$p(A \\mid d)$"); ax.legend(fontsize=8)

for col, (samples, color, label) in enumerate([
    (chain[BURNIN:, 1], "C1", "MCMC"), (fphys_valid[:, 1], "C2", "Flow")]):
    ax = axes[2, col]
    ax.hist(samples, bins=50, density=True, alpha=0.5, color=color, label=label)
    ax.plot(phi_vals, exact_margPhi, "C0", lw=1.5, label="exact")
    ax.axvline(TRUE_PHI, color="gray", ls="--", lw=0.7)
    ax.set_xlabel(r"$\varphi$"); ax.set_ylabel(r"$p(\varphi \mid d)$"); ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "sec5_mcmc_vs_flow.pdf", bbox_inches="tight")
plt.close(fig)


# ==================================================================
# Figure 5: Architecture comparison (Section 6)
# ==================================================================
print("Generating sec6_architecture_comparison.pdf (this takes several minutes) ...")
NL_ARCH, NE_ARCH = 12, 800

architectures = [
    ("Planar", init_planar, planar_forward, "C1", 5),
    ("Radial", init_radial, radial_forward, "C4", 4),
    ("Coupling", init_coupling, coupling_forward, "C2", 8),
]

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for col, (arch_name, init_fn, fwd_fn, color, ppl) in enumerate(architectures):
    print(f"  Training {arch_name}...")
    params, loss = train_flow(init_fn, fwd_fn, log_posterior,
                               n_layers=NL_ARCH, n_epochs=NE_ARCH,
                               lr_decay=0.0008)
    rng_a = np.random.default_rng(42)
    z_a = rng_a.normal(size=(3000, 2))
    fx_a, _ = fwd_fn(z_a, params)
    fp_a = flow_to_physical(fx_a)
    mask_a = ((fp_a[:, 0] >= A_RANGE[0]) & (fp_a[:, 0] <= A_RANGE[1]) &
              (fp_a[:, 1] >= PHI_RANGE[0]) & (fp_a[:, 1] <= PHI_RANGE[1]))
    valid_a = fp_a[mask_a]

    ax = axes[0, col]
    ax.contourf(A_vals, phi_vals, prob_grid, levels=15, cmap="viridis", alpha=0.25)
    if len(valid_a) > 0:
        ax.scatter(valid_a[:, 0], valid_a[:, 1], s=0.5, alpha=0.3, color=color, rasterized=True)
    ax.set_xlim(A_RANGE); ax.set_ylim(PHI_RANGE)
    ax.set_xlabel("$A$"); ax.set_ylabel(r"$\varphi$")
    ax.set_title(f"{arch_name} ({NL_ARCH}L · {NL_ARCH * ppl}p)", fontsize=10, color=color)
    ax.plot(TRUE_A, TRUE_PHI, "w+", markersize=8, mew=1)

    ax = axes[1, col]
    if len(valid_a) > 0: ax.hist(valid_a[:, 0], bins=45, density=True, alpha=0.5, color=color)
    ax.plot(A_vals, exact_margA, "C0", lw=1.3)
    ax.set_xlabel("$A$"); ax.set_ylabel("$p(A \\mid d)$")

    ax = axes[2, col]
    if len(valid_a) > 0: ax.hist(valid_a[:, 1], bins=45, density=True, alpha=0.5, color=color)
    ax.plot(phi_vals, exact_margPhi, "C0", lw=1.3)
    ax.set_xlabel(r"$\varphi$"); ax.set_ylabel(r"$p(\varphi \mid d)$")

fig.suptitle(f"Sinusoid — {NL_ARCH} layers, {NE_ARCH} epochs", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "sec6_architecture_comparison.pdf", bbox_inches="tight")
plt.close(fig)

print(f"\nDone. Figures saved to {OUT}/")
print("Files:")
for f in sorted(OUT.glob("*.pdf")):
    print(f"  {f}")
