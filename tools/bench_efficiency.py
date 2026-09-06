r"""
Step 1 / Stage A -- efficiency benchmark for the LoMix (EMCADNet) architecture.

Reproduces the efficiency claims of EMCAD / LoMix on local hardware:
parameter counts, MACs, inference latency, throughput and peak GPU memory.
Needs NO dataset and NO pretrained checkpoint, so it runs before data download.

Run from the LoMix repo root, e.g.:
    python ..\..\tools\bench_efficiency.py --encoders pvt_v2_b0 pvt_v2_b2 --img_size 224

Outputs: logs/bench_efficiency_<host>.json  and  a markdown table on stdout.
"""
import argparse, json, os, platform, statistics, sys, time
from datetime import datetime

import torch


def find_repo_root(start):
    d = os.path.abspath(start)
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "lib")) and os.path.isfile(os.path.join(d, "lib", "networks.py")):
            return d
        d = os.path.dirname(d)
    return None


def measure(model, x, iters, warmup, device):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x, mode="test")
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x, mode="test")
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
    return times, peak


def macs_of(model, img_size, device):
    """MACs via ptflops if available; returns None otherwise (optional dep)."""
    try:
        from ptflops import get_model_complexity_info
    except Exception:
        return None

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x, mode="test")[-1]

    try:
        with torch.no_grad():
            macs, _ = get_model_complexity_info(
                Wrap(model).to(device), (3, img_size, img_size),
                as_strings=False, print_per_layer_stat=False, verbose=False)
        return float(macs)
    except Exception as e:
        print(f"  [warn] ptflops failed: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="LoMix repo root (auto-detected from cwd)")
    ap.add_argument("--encoders", nargs="+", default=["pvt_v2_b0", "pvt_v2_b2"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_classes", type=int, default=9)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = args.repo or find_repo_root(os.getcwd()) or find_repo_root(__file__)
    if repo is None:
        sys.exit("Could not locate the LoMix repo root. Pass --repo <path to LoMix>.")
    sys.path.insert(0, repo)
    os.chdir(repo)
    from lib.networks import EMCADNet  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "capability": list(torch.cuda.get_device_capability(0)) if device.type == "cuda" else None,
        "vram_total_MiB": round(torch.cuda.get_device_properties(0).total_memory / 2**20) if device.type == "cuda" else None,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "num_classes": args.num_classes,
        "iters": args.iters,
        "warmup": args.warmup,
    }
    print(json.dumps(env, indent=2))

    rows = []
    for enc in args.encoders:
        print(f"\n=== {enc} ===")
        model = EMCADNet(num_classes=args.num_classes, encoder=enc, pretrain=False).to(device)
        n_enc = sum(p.numel() for p in model.backbone.parameters())
        n_dec = sum(p.numel() for p in model.decoder.parameters())
        n_all = sum(p.numel() for p in model.parameters())
        x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
        macs = macs_of(model, args.img_size, device)
        times, peak = measure(model, x, args.iters, args.warmup, device)
        row = {
            "encoder": enc,
            "params_encoder_M": round(n_enc / 1e6, 3),
            "params_decoder_M": round(n_dec / 1e6, 3),
            "params_total_M": round(n_all / 1e6, 3),
            "gmacs": round(macs / 1e9, 3) if macs else None,
            "latency_ms_mean": round(statistics.mean(times), 2),
            "latency_ms_std": round(statistics.pstdev(times), 2),
            "throughput_img_s": round(args.batch_size * 1000.0 / statistics.mean(times), 2),
            "peak_infer_vram_MiB": round(peak, 1),
        }
        rows.append(row)
        print(json.dumps(row, indent=2))
        del model, x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs",
                                   f"bench_efficiency_{platform.node()}.json")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"env": env, "rows": rows}, f, indent=2)

    hdr = ["encoder", "params_total_M", "params_decoder_M", "gmacs",
           "latency_ms_mean", "throughput_img_s", "peak_infer_vram_MiB"]
    print("\n| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        print("| " + " | ".join(str(r.get(h)) for h in hdr) + " |")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
