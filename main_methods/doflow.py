"""DoFlow adapter. Uses past cases and past treatment only."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from doflow_official_blocks.doflow_official import RNN, CNF


def train_loop(model, loss_fn, epochs, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best = float("inf"); state = None
    for _ in range(epochs):
        model.train(); opt.zero_grad(); loss = loss_fn()
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite training loss: {float(loss.detach())}")
        loss.backward(); grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(grad_norm): raise FloatingPointError(f"non-finite gradient norm: {float(grad_norm)}")
        opt.step()
        value = float(loss.detach())
        if value < best: best = value; state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if state is not None: model.load_state_dict(state)
    model.eval(); return best


def fit_sample(d, tr, ev, seed=0, steps=30, n_samples=100, lr=2e-3, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed); np.random.seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    origin, horizon = d["o"], d["H"]
    yf = d["y"][..., None].astype("float32"); af = d["A"][..., None].astype("float32")
    yl = np.log1p(np.maximum(yf, 0)); lev = yl[:, :origin, 0].mean(axis=1)[:, None]
    trm = np.zeros(len(yf), bool); trm[tr] = True
    scale = max(float((yl - lev[:, :, None])[trm, :origin].std()), 1e-8); yn = (yl - lev[:, :, None]) / scale
    yt, at = torch.tensor(yn, device=device), torch.tensor(af, device=device)
    trt, te = torch.tensor(tr, device=device), torch.tensor(ev, device=device)
    ry, ra = RNN(1, 16, 2).to(device), RNN(1, 16, 2).to(device); flow = CNF(horizon, 32, 64, 3).to(device); model = nn.ModuleList([ry, ra, flow]).to(device)
    yhist, ahist, target = yt[trt, :origin], at[trt, :origin], yt[trt, origin:, 0]
    def flow_loss():
        hy = ry(yhist)[1]; ha = ra(ahist)[1]; cond = torch.cat([hy, ha], 1); noise = torch.randn_like(target)
        s = torch.rand(len(target), 1, device=target.device); state = (1 - s) * noise + s * target
        return F.mse_loss(flow(s, torch.cat([state, cond], 1)), target - noise)
    train_loop(model, flow_loss, steps, lr)
    with torch.no_grad():
        cond = torch.cat([ry(yt[te, :origin])[1], ra(at[te, :origin])[1]], 1); k = n_samples
        draws = flow.sample(cond.repeat(k, 1), num_steps=30, forward=True)[0].view(k, len(cond), -1).cpu().numpy()
    S = np.maximum(np.expm1(draws * scale + lev[ev][None]), 0)
    return S
