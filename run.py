"""python run.py --method transformer|diffusion|doflow --dataset epicf|iraq|pm25|cancer --seed 0"""
import argparse, os, time, numpy as np, torch
import data as D


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=("transformer", "diffusion", "doflow"))
    p.add_argument("--dataset", required=True, choices=D.DATASETS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--origin", type=int, default=int(os.environ.get("SL_ORIGIN", 56)))
    p.add_argument("--steps", type=int, default=None,)
    p.add_argument("--n-samples", type=int, default=100)
    a = p.parse_args()
    d = D.setup(a.dataset, a.origin); tr, ev = D.split(a.dataset); o = d["o"]; truth = d["y"][ev, o:]
    t0 = time.time()
    if a.method == "doflow":
        from main_methods import doflow as M
        steps = a.steps if a.steps is not None else 30
        S = M.fit_sample(d, tr, ev, seed=a.seed, steps=steps, n_samples=a.n_samples); dev = "cuda" if torch.cuda.is_available() else "cpu"
        p_ = S.mean(0)
    else:
        from main_methods import transformer_diffusion as M
        steps = a.steps if a.steps is not None else int(os.environ.get("DIFF_STEPS", 2000)); dev = M.DEV
        S = M.growth_cap(d, tr, ev, M.fit_sample(d, tr, ev, seed=a.seed, mode="det" if a.method == "transformer" else "eps", train_steps=steps))
        p_ = np.median(S, 0)
    res = D.all_metrics(p_, truth, S if S.shape[0] > 1 else None)
    print(f"device={dev} method={a.method} dataset={a.dataset} seed={a.seed} origin={o} steps={steps} train_units={len(tr)} test_units={len(ev)} ({time.time()-t0:.0f}s)")
    print(res)


if __name__ == "__main__":
    main()
