import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    from functools import partial

    # Reproducibility
    rng_global = np.random.default_rng(42)

    # Plotting defaults
    plt.rcParams.update({
        "figure.facecolor": "#0c0c20",
        "axes.facecolor": "#0c0c20",
        "axes.edgecolor": "#2a3a4a",
        "axes.labelcolor": "#8098b0",
        "xtick.color": "#506878",
        "ytick.color": "#506878",
        "text.color": "#c8d8e8",
        "font.family": "monospace",
        "font.size": 10,
        "figure.dpi": 120,
    })
    return GridSpec, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    > **⚡ How this notebook works**
    >
    > This tutorial is built with [marimo](https://marimo.io), a reactive
    > Python notebook. A few things to know:
    >
    > - **Cells are reactive.** When you change a slider or edit a cell,
    >   every cell that depends on it **re-executes automatically** — like a
    >   spreadsheet. You don't need to manually re-run downstream cells.
    > - **Training cells require a click.** Sections with expensive training
    >   (Sections 4, 5, 6) have a **Run** button. Adjust the sliders first,
    >   then click the button to start training. This prevents long waits
    >   on startup or when tweaking settings.
    > - **You can edit any cell.** Click on a code cell to modify it — change
    >   a distribution, tweak a hyperparameter, add a plot. The notebook
    >   will re-execute everything that depends on your change.
    > - **To run this notebook:** `marimo edit tutorial.py`
    >   (or `marimo run tutorial.py` for a read-only app).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Normalizing Flows for Bayesian Inference

    **An interactive tutorial for physics graduate students**

    ---

    You know how to write down a posterior:

    $$p(\theta \mid d) = \frac{p(d \mid \theta)\, p(\theta)}{p(d)}$$

    You know how to sample from it with MCMC. This tutorial introduces a
    fundamentally different approach: **learn an invertible map** that transforms
    a simple Gaussian into the posterior, then generate independent samples
    by pushing Gaussian draws through the map.

    These are called **normalizing flows**, and they're increasingly used in
    gravitational-wave inference, cosmology, and particle physics.

    **Prerequisites:** Bayesian inference, MCMC (you've run a chain before),
    basic calculus and linear algebra.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The problem: sampling from posteriors is expensive

    Consider a simple inference problem. We observe noisy data from a sinusoidal signal:

    $$d(t) = A \sin(2\pi t + \varphi) + n(t), \qquad n \sim \mathcal{N}(0, \sigma^2)$$

    and want to infer the amplitude $A$ and phase $\varphi$. The posterior
    $p(A, \varphi \mid d)$ has a **curved ridge** — there's a degeneracy between
    amplitude and phase that creates a banana-shaped distribution.

    Let's set this up and see what the posterior looks like.
    """)
    return


@app.cell
def _(np):
    # Ground truth
    TRUE_A, TRUE_PHI = 2.2, 1.8
    SIGMA = 0.8
    N_DATA = 8

    # Generate data
    _rng = np.random.default_rng(77)
    t_obs = np.array([(i + 0.5) / N_DATA for i in range(N_DATA)])
    data_obs = TRUE_A * np.sin(2 * np.pi * t_obs + TRUE_PHI) + SIGMA * _rng.normal(size=N_DATA)

    # Parameter ranges
    A_RANGE = (0.2, 4.5)
    PHI_RANGE = (0.0, 2 * np.pi)

    def log_likelihood(A, phi):
        """Log-likelihood for sinusoidal model."""
        model = A * np.sin(2 * np.pi * t_obs + phi)
        return -np.sum((data_obs - model) ** 2) / (2 * SIGMA ** 2)

    def log_posterior(A, phi):
        """Log-posterior with uniform prior on [A_RANGE] x [PHI_RANGE]."""
        if A < A_RANGE[0] or A > A_RANGE[1] or phi < PHI_RANGE[0] or phi > PHI_RANGE[1]:
            return -np.inf
        return log_likelihood(A, phi)

    # Vectorized version for grid evaluation
    def log_posterior_grid(A_grid, phi_grid):
        """Evaluate log-posterior on a 2D meshgrid."""
        model = A_grid[:, :, None] * np.sin(2 * np.pi * t_obs[None, None, :] + phi_grid[:, :, None])
        residuals = data_obs[None, None, :] - model
        ll = -np.sum(residuals ** 2, axis=-1) / (2 * SIGMA ** 2)
        # Apply prior bounds
        mask = (
            (A_grid >= A_RANGE[0]) & (A_grid <= A_RANGE[1]) &
            (phi_grid >= PHI_RANGE[0]) & (phi_grid <= PHI_RANGE[1])
        )
        ll[~mask] = -np.inf
        return ll

    return (
        A_RANGE,
        PHI_RANGE,
        SIGMA,
        TRUE_A,
        TRUE_PHI,
        data_obs,
        log_posterior,
        log_posterior_grid,
        t_obs,
    )


@app.cell
def _(
    A_RANGE,
    PHI_RANGE,
    SIGMA,
    TRUE_A,
    TRUE_PHI,
    data_obs,
    log_posterior_grid,
    np,
    plt,
    t_obs,
):
    GRID_N = 100
    A_vals = np.linspace(A_RANGE[0], A_RANGE[1], GRID_N)
    phi_vals = np.linspace(PHI_RANGE[0], PHI_RANGE[1], GRID_N)
    A_grid, phi_grid = np.meshgrid(A_vals, phi_vals)
    lp_grid = log_posterior_grid(A_grid, phi_grid)
    prob_grid = np.exp(lp_grid - np.nanmax(lp_grid))

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: data
    _ax = _axes[0]
    _t_fine = np.linspace(0, 1, 200)
    _ax.plot(_t_fine, TRUE_A * np.sin(2 * np.pi * _t_fine + TRUE_PHI),
            color="#3cb8c8", lw=1.5, label="true signal")
    _ax.errorbar(t_obs, data_obs, yerr=SIGMA, fmt="o", color="#e8a060",
                markersize=5, capsize=3, label="data")
    _ax.set_xlabel("t")
    _ax.set_ylabel("d(t)")
    _ax.legend(fontsize=8, loc="upper right")
    _ax.set_title("Observed data", fontsize=11)

    # Right: posterior
    _ax = _axes[1]
    _ax.contourf(A_vals, phi_vals, prob_grid, levels=20, cmap="viridis")
    _ax.axvline(TRUE_A, color="white", ls="--", lw=0.8, alpha=0.4)
    _ax.axhline(TRUE_PHI, color="white", ls="--", lw=0.8, alpha=0.4)
    _ax.plot(TRUE_A, TRUE_PHI, "w+", markersize=10, mew=1.5)
    _ax.set_xlabel("Amplitude A")
    _ax.set_ylabel("Phase φ")
    _ax.set_title("Posterior p(A, φ | d)", fontsize=11)

    _fig.tight_layout()
    _fig
    return A_vals, GRID_N, phi_vals, prob_grid


@app.cell
def _(mo):
    mo.md(r"""
    The posterior has a **curved ridge** — the data are nearly as well fit by
    slightly increasing $A$ while shifting $\varphi$ to compensate. This
    amplitude–phase degeneracy is a preview of the kinds of correlations that
    appear in gravitational-wave parameter estimation (e.g., distance–inclination).

    MCMC handles this, but at a cost. A Metropolis-Hastings chain with isotropic
    proposals spends most of its time proposing steps *perpendicular* to the ridge,
    which get rejected. The result: highly correlated samples and an effective
    sample size (ESS) much smaller than the chain length.

    **Can we do better?** What if, instead of walking through the posterior one
    step at a time, we could learn a function that *maps* a simple distribution
    directly onto the posterior — and then generate as many independent samples
    as we want?

    ---

    ## 2. The core idea: change of variables

    The idea starts from something you already know: the **change-of-variables
    formula** from multivariate calculus. If you have a random variable $z$
    with known density $q_0(z)$ and you apply a smooth, invertible
    transformation to get $x$, you can write down the density of $x$
    exactly — you just account for how the transformation stretches or
    compresses volume.

    **Notation.** We write $q_0$ for the **base density** (a standard
    Gaussian $\mathcal{N}(0,I)$). We write $f_\theta$ for a parametric,
    invertible map that we will learn. And $q_\theta$ denotes the
    **pushforward density** — the distribution of $x = f_\theta(z)$ when
    $z \sim q_0$. Note that $q_0$ and $q_\theta$ are *different densities*
    in different spaces; they are related by the transformation $f_\theta$,
    not by a change of subscript.

    The change-of-variables formula then gives:

    $$q_\theta(x) = q_0\!\left(f_\theta^{-1}(x)\right) \left|\det \frac{\partial f_\theta^{-1}}{\partial x}\right|
    = q_0(z) \left|\det \frac{\partial f_\theta}{\partial z}\right|^{-1}$$

    The Jacobian determinant captures the local volume change. If you've
    seen Jacobians in coordinate transformations in GR or in changing
    integration variables for a partition function, this is the same idea.

    **The plan:** if we can find $\theta$ such that $q_\theta \approx p(\cdot \mid d)$:

    1. Choose a parametric family of invertible maps $f_\theta$
    2. Adjust $\theta$ until $q_\theta(x) \approx p(x \mid d)$
    3. Sample: draw $z \sim \mathcal{N}(0, I)$, compute $x = f_\theta(z)$, done

    Every sample is **independent** (no chain, no burn-in, no thinning), and
    the cost of generating each sample is just one forward pass through $f_\theta$.
    This is the essence of a **normalizing flow**. The name "normalizing" refers
    to the fact that the change-of-variables formula keeps the density properly
    normalized; "flow" evokes the smooth deformation of the base into the target.

    ---

    ## 3. Planar flows: the simplest architecture

    The simplest invertible layer is the **planar flow** (Rezende & Mohamed, 2015):

    $$f(z) = z + u \tanh(w^\top z + b)$$

    where $u, w \in \mathbb{R}^2$ and $b \in \mathbb{R}$ are learnable parameters —
    just **5 numbers** per layer. This adds a tanh "bump" in the direction $u$,
    with the bump's location and orientation controlled by $w$ and $b$.

    The Jacobian determinant has a closed-form expression:

    $$\left|\det \frac{\partial f}{\partial z}\right| = \left|1 + u^\top \text{diag}(1 - \tanh^2(w^\top z + b))\, w\right|$$

    which is a scalar — trivially cheap to compute.

    One layer can only do a single directional warp — the bump is always
    along $u$, so a single planar layer can't simultaneously stretch in one
    direction and compress in another. This is a real limitation: you need
    *many* layers even for modest 2D targets, and the learned directions
    $u_k$ of different layers need not be orthogonal or evenly spread.
    We'll see in Section 6 that more flexible layer designs largely remove
    this bottleneck.

    To build up expressiveness, we **compose** layers:

    $$f = f_K \circ f_{K-1} \circ \cdots \circ f_1$$

    The log-determinant of the composition is just the sum of the per-layer
    log-determinants (chain rule). More layers = more expressive, at a
    linear cost in computation.

    Let's implement this and see what it does to a Gaussian.
    """)
    return


@app.cell
def _(np):
    def init_planar(n_layers, seed=55):
        """Initialize planar flow parameters."""
        rng = np.random.default_rng(seed)
        return [
            {
                "u": rng.normal(0, 0.3, size=2),
                "w": rng.normal(0, 0.3, size=2),
                "b": rng.normal(0, 0.1),
            }
            for _ in range(n_layers)
        ]

    def planar_forward(z, params):
        """Push z through the planar flow. Returns transformed x and log|det J|."""
        x = z.copy()
        log_det = 0.0
        for layer in params:
            u, w, b = layer["u"], layer["w"], layer["b"]
            dot = x @ w + b                          # (N,)
            th = np.tanh(dot)                         # (N,)
            dth = 1 - th ** 2                         # dtanh/dx
            # Jacobian determinant for each sample
            psi = dth[:, None] * w[None, :]           # (N, 2)
            det_term = 1 + psi @ u                    # (N,)
            log_det = log_det + np.log(np.abs(det_term) + 1e-10)
            x = x + np.outer(th, u)                   # (N, 2)
        return x, log_det

    def planar_forward_single(z, params):
        """Forward pass for a single point (for visualization)."""
        x = z.copy()
        log_det = 0.0
        for layer in params:
            u, w, b = layer["u"], layer["w"], layer["b"]
            dot = w @ x + b
            th = np.tanh(dot)
            dth = 1 - th ** 2
            log_det += np.log(abs(1 + u @ (dth * w)) + 1e-10)
            x = x + u * th
        return x, log_det

    return init_planar, planar_forward


@app.cell
def _(mo):
    layer_slider = mo.ui.slider(1, 20, value=5, label="Number of planar layers", step=1)
    layer_slider
    return (layer_slider,)


@app.cell
def _(GridSpec, init_planar, layer_slider, np, planar_forward, plt):
    _n_layers = layer_slider.value
    _params = init_planar(_n_layers, seed=55)

    # Base Gaussian samples
    _rng = np.random.default_rng(42)
    _z = _rng.normal(size=(2000, 2))

    # Transform through increasing numbers of layers
    _n_show = min(_n_layers, 6)
    _layer_indices = np.linspace(0, _n_layers, _n_show + 1, dtype=int)

    _fig = plt.figure(figsize=(12, 3.5))
    _gs = GridSpec(1, _n_show + 1, figure=_fig, wspace=0.05)

    for _idx, _k in enumerate(_layer_indices):
        _ax = _fig.add_subplot(_gs[0, _idx])
        if _k == 0:
            _pts = _z
        else:
            _pts, _ = planar_forward(_z, _params[:_k])

        _ax.scatter(_pts[:, 0], _pts[:, 1], s=0.4, alpha=0.3, c="#70c8a0", rasterized=True)
        _ax.set_xlim(-4.5, 4.5)
        _ax.set_ylim(-4.5, 4.5)
        _ax.set_aspect("equal")
        if _idx == 0:
            _ax.set_title("Base z~N(0,I)", fontsize=9)
        else:
            _ax.set_title(f"After {_k} layer{'s' if _k > 1 else ''}", fontsize=9)
        _ax.tick_params(labelsize=6)
        if _idx > 0:
            _ax.set_yticklabels([])

    _fig.suptitle(f"Gaussian → {_n_layers} planar layers: watching the distribution deform",
                 fontsize=11, y=1.02)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Training: minimizing KL divergence

    We want the flow's output distribution $q_\theta$ to match the target
    posterior $p$. We minimize the forward KL divergence:

    $$D_\text{KL}(q_\theta \| p) = \mathbb{E}_{z \sim \mathcal{N}(0,I)} \left[
    -\log p(f_\theta(z)) - \log\left|\det \frac{\partial f_\theta}{\partial z}\right|
    \right] + \text{const}$$

    **Key insight:** every term is computable without samples from $p$.

    - $z \sim \mathcal{N}(0, I)$: we draw from the base Gaussian (trivial)
    - $f_\theta(z)$: we push $z$ through the flow (forward pass)
    - $\log p(f_\theta(z))$: we evaluate the **unnormalized** posterior at the output
      (the normalizing constant $p(d)$ is constant w.r.t. $\theta$ and drops out)
    - $\log|\det J|$: the flow architecture gives us this cheaply *for the
      layer types we use here* (rank-one or triangular Jacobians). This is
      **not free in general** — a naïve $d \times d$ determinant costs $O(d^3)$,
      and much of the art in designing flow architectures is choosing
      transformations whose Jacobian structure makes the determinant cheap

    There is **no training data** in the usual ML sense. We're not fitting to
    examples — we're optimizing the flow parameters so that its pushforward
    distribution matches a target that we can evaluate pointwise.

    Each training step: draw a batch of $z_i$, push through $f_\theta$, evaluate
    the loss, compute gradients, update $\theta$. An **epoch** is one such update.

    Let's train a flow on our sinusoidal posterior and watch it converge.

    > **Note on the KL direction (and how it can bite you).** $D_\text{KL}(q \| p)$
    > is **mode-seeking**: it penalizes the flow for placing mass where $p$ is
    > small, but imposes *no direct penalty* for regions where $p$ has mass but
    > $q$ does not. If the target has multiple modes, the flow may latch onto one
    > and ignore the rest — and nothing in the training loss tells you a mode was
    > missed. You only discover the problem by independent validation (a short
    > MCMC run, a p–p plot, or physical intuition about expected degeneracies).
    >
    > The reverse, $D_\text{KL}(p \| q)$, would be mode-covering but requires
    > samples from $p$ (which is what we're trying to get). In practice, one can
    > mitigate mode-seeking by using the flow as a proposal for MCMC (Section 8),
    > or by training with alternative divergences.
    """)
    return


@app.cell
def _(A_RANGE, PHI_RANGE, np):
    def flow_to_physical(x, ranges=None):
        """Map flow output to physical parameter space via sigmoid."""
        if ranges is None:
            ranges = [A_RANGE, PHI_RANGE]
        out = np.empty_like(x)
        for d in range(2):
            sig = 1 / (1 + np.exp(-np.clip(x[:, d], -10, 10)))
            out[:, d] = ranges[d][0] + (ranges[d][1] - ranges[d][0]) * sig
        return out

    def log_sigmoid_jac(x, ranges=None):
        """Log |d(sigmoid transform)/dx| for each sample."""
        if ranges is None:
            ranges = [A_RANGE, PHI_RANGE]
        total = np.zeros(x.shape[0])
        for d in range(2):
            xc = np.clip(x[:, d], -10, 10)
            sig = 1 / (1 + np.exp(-xc))
            total += np.log(sig * (1 - sig) + 1e-10) + np.log(ranges[d][1] - ranges[d][0])
        return total

    def compute_loss(z_batch, params, forward_fn, log_p_fn, ranges=None):
        """KL divergence loss for a batch of base samples."""
        x, log_det = forward_fn(z_batch, params)
        phys = flow_to_physical(x, ranges)
        log_sj = log_sigmoid_jac(x, ranges)

        log_p_vals = np.array([log_p_fn(phys[i, 0], phys[i, 1]) for i in range(len(z_batch))])
        loss = -log_det - log_sj - log_p_vals
        return np.mean(loss)

    def train_flow(init_fn, forward_fn, log_p_fn, n_layers, n_epochs,
                   batch_size=100, lr=0.02, lr_decay=0.001, seed=55, ranges=None):
        """Train a flow and return snapshots of parameters + loss history."""
        import copy

        params = init_fn(n_layers, seed=seed)
        rng = np.random.default_rng(999)
        loss_history = []
        snapshots = [(0, copy.deepcopy(params), None)]

        eps = 1e-4  # finite difference step

        for epoch in range(1, n_epochs + 1):
            current_lr = lr * np.exp(-epoch * lr_decay)
            z_batch = rng.normal(size=(batch_size, 2))

            base_loss = compute_loss(z_batch, params, forward_fn, log_p_fn, ranges)
            loss_history.append(base_loss)

            # Finite-difference gradients
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
                            grad = np.clip((lp - lm) / (2 * eps), -2, 2)
                            val[idx] -= current_lr * grad
                    elif isinstance(val, (float, np.floating)):
                        layer[key] = val + eps
                        lp = compute_loss(z_batch, params, forward_fn, log_p_fn, ranges)
                        layer[key] = val - eps
                        lm = compute_loss(z_batch, params, forward_fn, log_p_fn, ranges)
                        layer[key] = val
                        grad = np.clip((lp - lm) / (2 * eps), -2, 2)
                        layer[key] -= current_lr * grad

            if epoch % 20 == 0 or epoch == n_epochs:
                snapshots.append((epoch, copy.deepcopy(params), base_loss))

        return params, loss_history, snapshots

    return flow_to_physical, train_flow


@app.cell
def _(mo):
    train_layers_slider = mo.ui.slider(1, 20, value=8, label="Planar layers", step=1)
    train_epochs_slider = mo.ui.slider(100, 2000, value=800, label="Training epochs", step=100)
    train_run_btn = mo.ui.run_button(label="Train flow")
    mo.hstack([train_layers_slider, train_epochs_slider, train_run_btn], gap=2)
    return train_epochs_slider, train_layers_slider, train_run_btn


@app.cell
def _(
    A_RANGE,
    A_vals,
    GridSpec,
    PHI_RANGE,
    TRUE_A,
    TRUE_PHI,
    flow_to_physical,
    init_planar,
    log_posterior,
    mo,
    np,
    phi_vals,
    planar_forward,
    plt,
    prob_grid,
    train_epochs_slider,
    train_flow,
    train_layers_slider,
    train_run_btn,
):
    mo.stop(
        not train_run_btn.value,
        mo.md("*Adjust sliders above, then click **Train flow** to run.*"),
    )

    _nL = train_layers_slider.value
    _nE = train_epochs_slider.value

    # Train
    _trained_params, _loss_hist, _snapshots = train_flow(
        init_planar, planar_forward, log_posterior,
        n_layers=_nL, n_epochs=_nE,
        batch_size=100, lr=0.02, lr_decay=0.001, seed=55
    )

    # Sample from trained flow
    _rng = np.random.default_rng(42)
    _z_samples = _rng.normal(size=(3000, 2))
    _x_samples, _ = planar_forward(_z_samples, _trained_params)
    _phys_samples = flow_to_physical(_x_samples)

    # Figure: 3 panels — loss curve, exact posterior, flow samples
    _fig = plt.figure(figsize=(12, 4))
    _gs = GridSpec(1, 3, figure=_fig, wspace=0.35)

    # Loss curve
    _ax0 = _fig.add_subplot(_gs[0, 0])
    _ax0.plot(range(1, len(_loss_hist) + 1), _loss_hist, color="#70c8a0", lw=1)
    _ax0.set_xlabel("Epoch")
    _ax0.set_ylabel("Loss (KL divergence)")
    _ax0.set_title("Training loss", fontsize=11)
    if len(_loss_hist) > 50:
        _ymin = min(_loss_hist[50:])
        _ymax = max(_loss_hist[50:])
        _margin = (_ymax - _ymin) * 0.2
        _ax0.set_ylim(_ymin - _margin, _ymax + _margin)

    # Exact posterior
    _ax1 = _fig.add_subplot(_gs[0, 1])
    _ax1.contourf(A_vals, phi_vals, prob_grid, levels=20, cmap="viridis", alpha=0.8)
    _ax1.set_xlabel("A")
    _ax1.set_ylabel("φ")
    _ax1.set_title("Exact posterior", fontsize=11)
    if TRUE_A is not None:
        _ax1.plot(TRUE_A, TRUE_PHI, "w+", markersize=10, mew=1.5)

    # Flow samples
    _ax2 = _fig.add_subplot(_gs[0, 2])
    _ax2.contourf(A_vals, phi_vals, prob_grid, levels=20, cmap="viridis", alpha=0.2)
    _mask = (
        (_phys_samples[:, 0] >= A_RANGE[0]) & (_phys_samples[:, 0] <= A_RANGE[1]) &
        (_phys_samples[:, 1] >= PHI_RANGE[0]) & (_phys_samples[:, 1] <= PHI_RANGE[1])
    )
    _ax2.scatter(_phys_samples[_mask, 0], _phys_samples[_mask, 1], s=0.5, alpha=0.3,
                color="#70c8a0", rasterized=True)
    _ax2.set_xlim(A_RANGE)
    _ax2.set_ylim(PHI_RANGE)
    _ax2.set_xlabel("A")
    _ax2.set_ylabel("φ")
    _ax2.set_title(f"Flow samples ({_nL}L, {_nE} epochs)", fontsize=11)
    if TRUE_A is not None:
        _ax2.plot(TRUE_A, TRUE_PHI, "w+", markersize=10, mew=1.5)

    _fig.suptitle("Training a planar flow on the sinusoidal posterior", fontsize=12, y=1.02)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. MCMC vs normalizing flows: head to head

    Now let's compare the two approaches directly on the same posterior.

    **MCMC (Metropolis-Hastings):** cost is at *inference time*. Each step
    proposes a move, evaluates $p$, and accepts or rejects. Samples are
    correlated, so the effective sample size (ESS) is smaller than the chain
    length. More samples = more time.

    **Normalizing flow:** cost is *upfront* during training. Once trained,
    sampling is nearly free — draw $z \sim \mathcal{N}(0,I)$, push through $f$,
    get an independent posterior sample. ESS $= N$ by construction.

    However, the training cost depends strongly on the architecture: a toy
    planar flow trains in seconds, while a production neural spline flow can
    require hours of GPU time. For a one-off inference on a single event, the
    upfront cost may exceed MCMC; the payoff comes with amortization or when
    many independent samples are needed quickly.

    Adjust the sliders to explore the tradeoff.
    """)
    return


@app.cell
def _(log_posterior, np):
    def run_mcmc(n_steps, prop_scale=0.25, seed=123, start=None):
        """Run Metropolis-Hastings and return chain + diagnostics."""
        rng = np.random.default_rng(seed)
        if start is None:
            start = np.array([1.0, 4.0])
        chain = np.empty((n_steps + 1, 2))
        chain[0] = start
        current_lp = log_posterior(start[0], start[1])
        n_accept = 0

        for i in range(n_steps):
            proposal = chain[i] + prop_scale * rng.normal(size=2)
            prop_lp = log_posterior(proposal[0], proposal[1])
            if np.log(rng.uniform()) < prop_lp - current_lp:
                chain[i + 1] = proposal
                current_lp = prop_lp
                n_accept += 1
            else:
                chain[i + 1] = chain[i]

        return chain, n_accept / n_steps

    def compute_ess(chain, burnin=200):
        """Effective sample size from autocorrelation of first parameter."""
        vals = chain[burnin:, 0]
        n = len(vals)
        if n < 20:
            return 0
        mean = np.mean(vals)
        c0 = np.mean((vals - mean) ** 2)
        if c0 == 0:
            return n
        tau_sum = 0
        for lag in range(1, min(n, 500)):
            ck = np.mean((vals[:-lag] - mean) * (vals[lag:] - mean))
            rho = ck / c0
            if rho < 0.05:
                break
            tau_sum += rho
        return max(1, int(n / (1 + 2 * tau_sum)))

    return compute_ess, run_mcmc


@app.cell
def _(mo):
    mcmc_steps_slider = mo.ui.slider(500, 10000, value=5000, label="MCMC steps", step=500)
    flow_layers_compare = mo.ui.slider(1, 20, value=8, label="Flow layers", step=1)
    flow_epochs_compare = mo.ui.slider(200, 2000, value=800, label="Flow epochs", step=100)
    compare_run_btn = mo.ui.run_button(label="Run comparison")
    mo.hstack([mcmc_steps_slider, flow_layers_compare, flow_epochs_compare, compare_run_btn], gap=2)
    return compare_run_btn, flow_epochs_compare, flow_layers_compare, mcmc_steps_slider


@app.cell
def _(
    A_RANGE,
    A_vals,
    GRID_N,
    GridSpec,
    PHI_RANGE,
    TRUE_A,
    TRUE_PHI,
    compare_run_btn,
    compute_ess,
    flow_epochs_compare,
    flow_layers_compare,
    flow_to_physical,
    init_planar,
    log_posterior,
    mcmc_steps_slider,
    mo,
    np,
    phi_vals,
    planar_forward,
    plt,
    prob_grid,
    run_mcmc,
    train_flow,
):
    mo.stop(
        not compare_run_btn.value,
        mo.md("*Adjust sliders above, then click **Run comparison** to run.*"),
    )

    _n_mcmc = mcmc_steps_slider.value
    _nL = flow_layers_compare.value
    _nE = flow_epochs_compare.value
    _burnin = 200

    # Run MCMC
    _chain, _acc_rate = run_mcmc(_n_mcmc)
    _ess = compute_ess(_chain, _burnin)

    # Train flow
    _params, _loss, _ = train_flow(
        init_planar, planar_forward, log_posterior,
        n_layers=_nL, n_epochs=_nE,
        batch_size=100, lr=0.02, lr_decay=0.001
    )

    # Flow samples
    _rng = np.random.default_rng(42)
    _z = _rng.normal(size=(3000, 2))
    _fx, _ = planar_forward(_z, _params)
    _fphys = flow_to_physical(_fx)
    _fmask = (
        (_fphys[:, 0] >= A_RANGE[0]) & (_fphys[:, 0] <= A_RANGE[1]) &
        (_fphys[:, 1] >= PHI_RANGE[0]) & (_fphys[:, 1] <= PHI_RANGE[1])
    )
    _fphys_valid = _fphys[_fmask]

    # Marginals from exact grid
    _dA = (A_RANGE[1] - A_RANGE[0]) / GRID_N
    _dP = (PHI_RANGE[1] - PHI_RANGE[0]) / GRID_N
    _exact_margA = np.sum(prob_grid, axis=0)
    _exact_margA = _exact_margA / (np.sum(_exact_margA) * _dA)
    _exact_margPhi = np.sum(prob_grid, axis=1)
    _exact_margPhi = _exact_margPhi / (np.sum(_exact_margPhi) * _dP)

    # ---- Plot ----
    _fig = plt.figure(figsize=(12, 8))
    _gs = GridSpec(3, 2, figure=_fig, hspace=0.4, wspace=0.3)

    # Top row: 2D panels
    for _col, (_title, _samples, _color, _stats_text) in enumerate([
        ("MCMC (Metropolis-Hastings)",
         _chain[_burnin:],
         "#e8a060",
         f"steps: {_n_mcmc}\nsamples: {_n_mcmc - _burnin}\nESS: {_ess}\naccept: {_acc_rate:.0%}"),
        (f"Normalizing Flow ({_nL}L, {_nE} epochs)",
         _fphys_valid,
         "#70c8a0",
         f"samples: {len(_fphys_valid)}\nESS: {len(_fphys_valid)}\n(all independent)"),
    ]):
        _ax = _fig.add_subplot(_gs[0, _col])
        _ax.contourf(A_vals, phi_vals, prob_grid, levels=15, cmap="viridis", alpha=0.25)
        _ax.scatter(_samples[:, 0], _samples[:, 1], s=0.4, alpha=0.25,
                   color=_color, rasterized=True)
        _ax.set_xlim(A_RANGE)
        _ax.set_ylim(PHI_RANGE)
        _ax.set_xlabel("A")
        _ax.set_ylabel("φ")
        _ax.set_title(_title, fontsize=10, color=_color)
        _ax.text(0.97, 0.97, _stats_text, transform=_ax.transAxes,
                fontsize=8, va="top", ha="right", color="#8098b0",
                family="monospace")
        if TRUE_A is not None:
            _ax.plot(TRUE_A, TRUE_PHI, "w+", markersize=8, mew=1)

    # Middle row: marginal A
    for _col, (_samples, _color, _label) in enumerate([
        (_chain[_burnin:, 0], "#e8a060", "MCMC"),
        (_fphys_valid[:, 0], "#70c8a0", "Flow"),
    ]):
        _ax = _fig.add_subplot(_gs[1, _col])
        _ax.hist(_samples, bins=50, density=True, alpha=0.5, color=_color, label=_label)
        _ax.plot(A_vals, _exact_margA, color="#3cb8c8", lw=1.5, label="exact")
        if TRUE_A is not None:
            _ax.axvline(TRUE_A, color="white", ls="--", lw=0.7, alpha=0.4)
        _ax.set_xlabel("A")
        _ax.set_ylabel("p(A | d)")
        _ax.legend(fontsize=8)

    # Bottom row: marginal phi
    for _col, (_samples, _color, _label) in enumerate([
        (_chain[_burnin:, 1], "#e8a060", "MCMC"),
        (_fphys_valid[:, 1], "#70c8a0", "Flow"),
    ]):
        _ax = _fig.add_subplot(_gs[2, _col])
        _ax.hist(_samples, bins=50, density=True, alpha=0.5, color=_color, label=_label)
        _ax.plot(phi_vals, _exact_margPhi, color="#3cb8c8", lw=1.5, label="exact")
        if TRUE_PHI is not None:
            _ax.axvline(TRUE_PHI, color="white", ls="--", lw=0.7, alpha=0.4)
        _ax.set_xlabel("φ")
        _ax.set_ylabel("p(φ | d)")
        _ax.legend(fontsize=8)

    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Architecture matters

    Planar flows are pedagogically useful but limited — each layer only warps
    along one direction. Let's compare three architectures:

    **Planar:** $z' = z + u \tanh(w^\top z + b)$ — 5 params/layer.
    Cheap but weak: one directional warp per layer.

    **Radial:** $z' = z + \beta\, h(\alpha, r)(z - z_0)$ — 4 params/layer.
    Expands or contracts around a learned center. Good for changing spread,
    bad for correlations.

    **Affine coupling:** Hold one dimension fixed, apply a learned scale and
    shift to the other:
    $x_b = z_b \cdot e^{s(z_a)} + t(z_a)$
    where $s$ and $t$ are parameterized functions of $z_a$ — 8 params/layer.
    Alternates which dimension is held fixed. This is the baby version of
    RealNVP and the architecture used in production tools like DINGO.

    Try different distributions below to see where each architecture
    succeeds and fails.
    """)
    return


@app.cell
def _(np):
    def init_radial(n_layers, seed=55):
        rng = np.random.default_rng(seed)
        return [
            {
                "z0": rng.normal(0, 0.3, size=2),
                "log_alpha": rng.normal(0, 0.1),
                "beta": rng.normal(0, 0.2),
            }
            for _ in range(n_layers)
        ]

    def radial_forward(z, params):
        x = z.copy()
        log_det = np.zeros(len(z))
        for layer in params:
            z0, log_alpha, beta = layer["z0"], layer["log_alpha"], layer["beta"]
            alpha = np.exp(log_alpha) + 0.01
            diff = x - z0[None, :]
            r = np.sqrt(np.sum(diff ** 2, axis=1)) + 1e-8
            h = 1.0 / (alpha + r)
            hp = -h ** 2
            scale = 1 + beta * h
            det = scale * (scale + r * beta * hp)
            log_det += np.log(np.abs(det) + 1e-10)
            x = x + beta * (h[:, None] * diff)
        return x, log_det

    def init_coupling(n_layers, seed=55):
        rng = np.random.default_rng(seed)
        return [
            {
                "dim": i % 2,
                "sa1": rng.normal(0, 0.3), "sa2": 0.5 + rng.normal(0, 0.2),
                "sa3": rng.normal(0, 0.1), "sa4": rng.normal(0, 0.1),
                "tb1": rng.normal(0, 0.3), "tb2": 0.5 + rng.normal(0, 0.2),
                "tb3": rng.normal(0, 0.1), "tb4": rng.normal(0, 0.1),
            }
            for i in range(n_layers)
        ]

    def coupling_forward(z, params):
        x = z.copy()
        log_det = np.zeros(len(z))
        for L in params:
            d = L["dim"]
            other = 1 - d
            cond = x[:, d]
            log_s = L["sa1"] * np.tanh(L["sa2"] * cond + L["sa3"]) + L["sa4"]
            t = L["tb1"] * np.tanh(L["tb2"] * cond + L["tb3"]) + L["tb4"]
            x[:, other] = x[:, other] * np.exp(log_s) + t
            log_det += log_s
        return x, log_det

    return coupling_forward, init_coupling, init_radial, radial_forward


@app.cell
def _(mo):
    dist_selector = mo.ui.dropdown(
        options={
            "Sinusoid (A, φ)": "sinusoid",
            "Banana (Rosenbrock)": "banana",
            "Triple mode": "triplemode",
            "Ring": "ring",
            "Funnel (Neal)": "funnel",
        },
        value="Sinusoid (A, φ)",
        label="Target distribution",
    )
    arch_layers_slider = mo.ui.slider(3, 20, value=12, label="Layers per architecture", step=1)
    arch_epochs_slider = mo.ui.slider(200, 3000, value=800, label="Training epochs", step=200)
    arch_run_btn = mo.ui.run_button(label="Train all architectures")
    mo.hstack([dist_selector, arch_layers_slider, arch_epochs_slider, arch_run_btn], gap=2)
    return arch_epochs_slider, arch_layers_slider, arch_run_btn, dist_selector


@app.cell
def _(log_posterior, np):
    test_distributions = {
        "sinusoid": {
            "name": "Sinusoid",
            "labels": ["A", "φ"],
            "ranges": [(0.2, 4.5), (0.0, 2 * np.pi)],
            "true_vals": [2.2, 1.8],
            "log_p": log_posterior,
        },
        "banana": {
            "name": "Banana",
            "labels": ["x₁", "x₂"],
            "ranges": [(-2.5, 4.5), (-2.0, 12.0)],
            "true_vals": [1.0, 1.0],
            "log_p": lambda x, y: (
                -np.inf if x < -2.5 or x > 4.5 or y < -2 or y > 12
                else -((x - 1) ** 2) / 2 - 4 * ((y - x * x) ** 2)
            ),
        },
        "triplemode": {
            "name": "Triple mode",
            "labels": ["x₁", "x₂"],
            "ranges": [(-4.0, 6.0), (-4.0, 6.0)],
            "true_vals": None,
            "log_p": lambda x, y: (
                -np.inf if x < -4 or x > 6 or y < -4 or y > 6
                else np.log(
                    np.exp(-0.5 * (1.2*x*x - 0.8*x*y + 1.2*y*y))
                    + 0.8 * np.exp(-0.5 * (1.5*(x-2.5)**2 + 1.0*(x-2.5)*(y-2.5) + 1.5*(y-2.5)**2))
                    + 0.6 * np.exp(-0.5 * (0.6*(x-1)**2 + 0.6*(y+1.5)**2))
                    + 1e-30
                )
            ),
        },
        "ring": {
            "name": "Ring",
            "labels": ["x₁", "x₂"],
            "ranges": [(-5.5, 5.5), (-5.5, 5.5)],
            "true_vals": None,
            "log_p": lambda x, y: (
                -np.inf if x < -5.5 or x > 5.5 or y < -5.5 or y > 5.5
                else -((np.sqrt(x*x + y*y) - 3) ** 2) / (2 * 0.35 ** 2)
            ),
        },
        "funnel": {
            "name": "Funnel",
            "labels": ["x₁", "x₂"],
            "ranges": [(-8.0, 8.0), (-4.0, 4.0)],
            "true_vals": None,
            "log_p": lambda x, v: (
                -np.inf if x < -8 or x > 8 or v < -4 or v > 4
                else -(v ** 2) / 18 - 0.5 * np.log(2 * np.pi * np.exp(v)) - x ** 2 / (2 * np.exp(v))
            ),
        },
    }
    return (test_distributions,)


@app.cell
def _(
    GridSpec,
    arch_epochs_slider,
    arch_layers_slider,
    arch_run_btn,
    coupling_forward,
    dist_selector,
    flow_to_physical,
    init_coupling,
    init_planar,
    init_radial,
    mo,
    np,
    planar_forward,
    plt,
    radial_forward,
    test_distributions,
    train_flow,
):
    mo.stop(
        not arch_run_btn.value,
        mo.md("*Choose a distribution and adjust sliders above, then click **Train all architectures** to run.*"),
    )

    _dist_key = dist_selector.value
    _dist = test_distributions[_dist_key]
    _nL = arch_layers_slider.value
    _nE = arch_epochs_slider.value
    _ranges = _dist["ranges"]
    _log_p = _dist["log_p"]

    # Compute exact posterior grid for this distribution
    _GN = 80
    _v0 = np.linspace(_ranges[0][0], _ranges[0][1], _GN)
    _v1 = np.linspace(_ranges[1][0], _ranges[1][1], _GN)
    _g0, _g1 = np.meshgrid(_v0, _v1)
    _lp = np.array([[_log_p(_g0[j, i], _g1[j, i]) for i in range(_GN)] for j in range(_GN)])
    _lp[~np.isfinite(_lp)] = -1e10
    _pgrid = np.exp(_lp - np.max(_lp))

    # Marginals
    _d0 = (_ranges[0][1] - _ranges[0][0]) / _GN
    _d1 = (_ranges[1][1] - _ranges[1][0]) / _GN
    _em0 = np.sum(_pgrid, axis=0); _em0 /= np.sum(_em0) * _d0
    _em1 = np.sum(_pgrid, axis=1); _em1 /= np.sum(_em1) * _d1

    _architectures = [
        ("Planar", init_planar, planar_forward, "#e8a060", 5),
        ("Radial", init_radial, radial_forward, "#c080e0", 4),
        ("Coupling", init_coupling, coupling_forward, "#70c8a0", 8),
    ]

    _fig = plt.figure(figsize=(13, 9))
    _gs = GridSpec(3, 3, figure=_fig, hspace=0.35, wspace=0.3)

    for _col, (_arch_name, _init_fn, _fwd_fn, _color, _ppl) in enumerate(_architectures):
        # Train
        _params, _loss, _ = train_flow(
            _init_fn, _fwd_fn, _log_p,
            n_layers=_nL, n_epochs=_nE,
            batch_size=100, lr=0.02, lr_decay=0.0008,
            ranges=_ranges,
        )

        # Sample
        _rng = np.random.default_rng(42)
        _z = _rng.normal(size=(3000, 2))
        _fx, _ = _fwd_fn(_z, _params)
        _fphys = flow_to_physical(_fx, _ranges)
        _mask = (
            (_fphys[:, 0] >= _ranges[0][0]) & (_fphys[:, 0] <= _ranges[0][1]) &
            (_fphys[:, 1] >= _ranges[1][0]) & (_fphys[:, 1] <= _ranges[1][1])
        )
        _valid = _fphys[_mask]

        # 2D
        _ax = _fig.add_subplot(_gs[0, _col])
        _ax.contourf(_v0, _v1, _pgrid, levels=15, cmap="viridis", alpha=0.25)
        if len(_valid) > 0:
            _ax.scatter(_valid[:, 0], _valid[:, 1], s=0.5, alpha=0.3, color=_color, rasterized=True)
        _ax.set_xlim(_ranges[0])
        _ax.set_ylim(_ranges[1])
        _ax.set_xlabel(_dist["labels"][0])
        _ax.set_ylabel(_dist["labels"][1])
        _ax.set_title(f"{_arch_name} ({_nL}L · {_nL * _ppl}p)", fontsize=10, color=_color)
        if _dist["true_vals"] is not None:
            _ax.plot(_dist["true_vals"][0], _dist["true_vals"][1], "w+", markersize=8, mew=1)

        # Marginal 0
        _ax = _fig.add_subplot(_gs[1, _col])
        if len(_valid) > 0:
            _ax.hist(_valid[:, 0], bins=45, density=True, alpha=0.5, color=_color)
        _ax.plot(_v0, _em0, color="#3cb8c8", lw=1.3)
        _ax.set_xlabel(_dist["labels"][0])
        _ax.set_ylabel(f"p({_dist['labels'][0]}|d)")
        if _dist["true_vals"] is not None:
            _ax.axvline(_dist["true_vals"][0], color="white", ls="--", lw=0.7, alpha=0.3)

        # Marginal 1
        _ax = _fig.add_subplot(_gs[2, _col])
        if len(_valid) > 0:
            _ax.hist(_valid[:, 1], bins=45, density=True, alpha=0.5, color=_color)
        _ax.plot(_v1, _em1, color="#3cb8c8", lw=1.3)
        _ax.set_xlabel(_dist["labels"][1])
        _ax.set_ylabel(f"p({_dist['labels'][1]}|d)")
        if _dist["true_vals"] is not None:
            _ax.axvline(_dist["true_vals"][1], color="white", ls="--", lw=0.7, alpha=0.3)

    _fig.suptitle(f"{_dist['name']} — {_nL} layers, {_nE} epochs",
                 fontsize=12, y=1.01)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. From toy models to gravitational-wave inference

    Everything we've built here uses **2D** posteriors with **5–8 parameters
    per layer** and **finite-difference gradients**. Real applications differ
    in scale but not in principle:

    - **DINGO** (Dax et al. 2021) uses normalizing flows for compact binary
      coalescence parameter estimation — 15+ dimensional posteriors with
      coupling layers whose conditioners are deep ResNets (thousands of
      parameters per layer), trained with automatic differentiation on GPUs.
      Note: DINGO uses a **discrete** flow (a finite composition of coupling
      layers), not a continuous normalizing flow. The use of AD for computing
      gradients during *training* doesn't make the flow itself continuous — AD
      replaces our finite-difference hack with exact backprop, but $f_\theta$
      is still a fixed sequence of discrete layers.

    - **Overlapping signals** (Langendorff et al. 2023): 3G detectors will
      have overlapping CBC signals. Joint PE is preferable to hierarchical
      analysis but expensive with traditional samplers. They demonstrate
      *conditional continuous normalizing flows* for this: the flow is
      *conditional* ($f_\theta(z; d)$ depends on the data, so one network
      handles many datasets) and *continuous* (neural ODE, not discrete
      layers — small memory footprint). Results: slightly widened posteriors,
      but injected values always recovered, and inference is orders of
      magnitude faster after training.

    - **Population inference** (Cheung et al. 2022): flows as likelihood
      emulators for GW population inference. Flows recover population
      posteriors using up to 300 mock injections where Gaussian process
      regression fails in high dimensions. Caveat: can underestimate
      uncertainty on real data.

    - **nessai** (Williams et al. 2021) combines flows with nested sampling,
      using the flow as a proposal distribution to accelerate evidence
      computation.

    - The key architectural upgrade is in the **conditioner**: our coupling
      layers use $s(z_a) = a_1 \tanh(a_2 z_a + a_3) + a_4$ (4 parameters).
      Production flows use $s(z_a) = \text{ResNet}(z_a)$ with tens of thousands
      of parameters, or neural spline flows where the transformation is a
      monotonic rational-quadratic spline.

    ## 8. Open questions

    - **Mode coverage:** $D_\text{KL}(q \| p)$ training is mode-seeking.
      How do you verify the flow hasn't missed a mode? (p-p plots, comparison
      with short MCMC runs, training with alternative objectives)

    - **Combining flows + MCMC:** Use the flow as a proposal distribution for
      Metropolis-Hastings: draw $z \sim \mathcal{N}(0,I)$, propose
      $x' = f_\theta(z)$, accept/reject via the usual ratio using the *true*
      posterior. Because the flow already approximates $p$, acceptance rates
      are high ($> 80\%$) and the chain mixes rapidly. This also provides a
      self-consistency check: if the chain discovers modes the flow missed,
      the acceptance rate drops, flagging the problem. This is arguably the
      most practical near-term use of flows in GW inference.

    - **Architecture taxonomy:** Beyond what we cover here, the main families
      are: *elementwise* (diagonal Jacobian, no correlations), *linear*
      (restrict $A$ to triangular/orthogonal for tractability), *autoregressive*
      (condition each dimension on all preceding — more expressive per layer
      than coupling, but sequential sampling), *residual* ($g(x) = x + F(x)$
      with Lipschitz constraint), and *continuous* (neural ODE
      $\frac{dx}{dt} = v_\theta(x, t)$, trace replaces determinant via
      Hutchinson estimator). Each trades off expressiveness, inversion cost,
      and Jacobian cost differently.

    - **Simulation-based inference:** When the likelihood is intractable
      (e.g., you can simulate but can't write down $p(d|\theta)$), flows
      can be trained on simulated $(\theta, d)$ pairs to learn the conditional
      posterior amortized over many datasets.

    - **Further open problems:** choice of base distribution (not always
      Gaussian); alternative loss functions beyond KL; flows on non-Euclidean
      manifolds (e.g., sky location on $S^2$); flows for discrete
      distributions.

    ---

    *Tutorial by Chiara Mingarelli, Abigail Moran, and Nicole Khusid
    (Yale / Flatiron CCA). Built with [marimo](https://marimo.io).*
    """)
    return


if __name__ == "__main__":
    app.run()
