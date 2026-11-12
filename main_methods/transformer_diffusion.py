"""Our transformer (mode="det") and our diffusion (mode="eps")."""
import os, numpy as np, torch, torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NSTEP, NSAMP, BATCH = 100, 32, 256


def windows(d, units, origins, with_target=True):
    y, A, S, T, H, Lw, k = (d[x] for x in ("y", "A", "S", "T", "H", "Lw", "k")); out = {"h": [], "a": [], "g": [], "r": [], "w": [], "lev": []}
    for t0 in origins:
        if with_target and t0 + H > T: continue
        lev = y[units, t0 - k:t0].mean(1) + 1.0; L = np.log(lev)[:, None]
        out["h"].append(np.log1p(np.maximum(y[units, t0 - Lw:t0], 0)) - L); out["a"].append(A[units, t0 - Lw:t0 + H])
        out["g"].append(np.concatenate([S[units], L, np.full((len(units), 1), t0 / T)], 1)); out["lev"].append(lev)
        if with_target:
            out["r"].append(np.log1p(np.maximum(y[units, t0:t0 + H], 0)) - L)
            out["w"].append(lev ** 2 / np.maximum(np.ptp(y[units, t0:t0 + H], 1), 1.0) ** 2)
    return {k_: np.concatenate(v) for k_, v in out.items() if v}


class Net(nn.Module):
    def __init__(self, Lw, H, gdim, dm=64, layers=3, heads=4):
        super().__init__()
        self.Lw, self.H = Lw, H; self.tok = nn.Linear(3, dm); self.pos = nn.Parameter(torch.randn(Lw + H, dm) * 0.02)
        self.cond = nn.Sequential(nn.Linear(gdim, dm), nn.SiLU(), nn.Linear(dm, dm)); self.temb = nn.Sequential(nn.Linear(16, dm), nn.SiLU(), nn.Linear(dm, dm))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, heads, 4 * dm, 0.1, batch_first=True, norm_first=True), layers)
        self.out = nn.Linear(dm, 1)

    def forward(self, h, xt, a, g, t):
        B = h.shape[0]; val = torch.cat([h, xt], 1); fut = torch.cat([torch.zeros(B, self.Lw, device=h.device), torch.ones(B, self.H, device=h.device)], 1)
        z = self.tok(torch.stack([val, a, fut], -1)) + self.pos[None]
        f = t[:, None].float() / NSTEP * torch.exp(torch.arange(8, device=h.device) * -1.0)[None]
        z = z + (self.cond(g) + self.temb(torch.cat([torch.sin(f), torch.cos(f)], 1)))[:, None]
        return self.out(self.enc(z))[:, self.Lw:, 0]


def schedule():
    s = torch.linspace(0, 1, NSTEP + 1); ab = torch.cos((s + 0.008) / 1.008 * np.pi / 2) ** 2; ab = ab / ab[0]
    beta = (1 - ab[1:] / ab[:-1]).clamp(max=0.999); alpha = 1 - beta; return beta.to(DEV), alpha.to(DEV), torch.cumprod(alpha, 0).to(DEV)


def fit_sample(d, tr, ev, seed=0, mode="eps", train_steps=2000):
    DET, EPS, TRAIN_STEPS = mode == "det", mode == "eps", train_steps
    torch.manual_seed(seed); np.random.seed(seed); o = d["o"]
    W = windows(d, tr, range(d["Lw"], o + 1, d["step"])); E = windows(d, ev, [o], with_target=False)
    hm, hs = W["h"].mean(), W["h"].std() + 1e-6; rm = W["r"].mean(0); rs = W["r"].std() + 1e-6; gm, gs = W["g"].mean(0), W["g"].std(0) + 1e-6
    T_ = lambda x: torch.tensor(x, dtype=torch.float32, device=DEV)
    h, a, g, r, w = T_((W["h"] - hm) / hs), T_(W["a"]), T_((W["g"] - gm) / gs), T_((W["r"] - rm) / rs), T_(W["w"] / W["w"].mean())
    net = Net(d["Lw"], d["H"], g.shape[1]).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TRAIN_STEPS); beta, alpha, ab = schedule(); n = len(h)
    for _ in range(TRAIN_STEPS):
        i = torch.randint(0, n, (min(BATCH, n),), device=DEV); t = torch.randint(0, NSTEP, (len(i),), device=DEV); eps = torch.randn_like(r[i])
        if DET:
            z0 = torch.zeros_like(r[i]); loss = (w[i][:, None] * (net(h[i], z0, a[i], g[i], torch.zeros_like(t)) - r[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); continue
        xt = ab[t].sqrt()[:, None] * r[i] + (1 - ab[t]).sqrt()[:, None] * eps
        loss = (w[i][:, None] * (net(h[i], xt, a[i], g[i], t) - (eps if EPS else r[i])) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); m = len(ev)
    he, ae, ge = (T_((E["h"] - hm) / hs).repeat(NSAMP, 1), T_(E["a"]).repeat(NSAMP, 1), T_((E["g"] - gm) / gs).repeat(NSAMP, 1))
    if DET:
        with torch.no_grad(): P = net(T_((E["h"] - hm) / hs), torch.zeros(m, d["H"], device=DEV), T_(E["a"]), T_((E["g"] - gm) / gs), torch.zeros(m, dtype=torch.long, device=DEV)).cpu().numpy()
        R = (P * rs + rm)[None]; cap = W["r"].max() + np.log(2)
        return np.maximum(E["lev"][None, :, None] * np.exp(np.minimum(R, cap)) - 1, 0)
    with torch.no_grad():
        x = torch.randn(m * NSAMP, d["H"], device=DEV)
        for t in reversed(range(NSTEP)):
            tt = torch.full((len(x),), t, device=DEV); out = net(he, x, ae, ge, tt)
            x0 = ((x - (1 - ab[t]).sqrt() * out) / ab[t].sqrt() if EPS else out).clamp(-5, 5)
            if t == 0: x = x0; break
            c0 = ab[t - 1].sqrt() * beta[t] / (1 - ab[t]); ct = alpha[t].sqrt() * (1 - ab[t - 1]) / (1 - ab[t])
            var = beta[t] * (1 - ab[t - 1]) / (1 - ab[t]); x = c0 * x0 + ct * x + var.sqrt() * torch.randn_like(x)
    R = x.cpu().numpy().reshape(NSAMP, m, d["H"]) * rs + rm; cap = W["r"].max() + np.log(2)
    S = np.maximum(E["lev"][None, :, None] * np.exp(np.minimum(R, cap)) - 1, 0)
    return S


def growth_cap(d, tr, ev, S):
    o, y = d["o"], d["y"]
    q = np.quantile((y[tr, o:].max(1) + 1) / (y[tr, :o].max(1) + 1), .99)
    return np.minimum(S, ((y[ev, :o].max(1) + 1) * q)[None, :, None])
