r"""
Step 1 / Stage B -- find the largest training configuration that fits this GPU.

Runs a real forward + LoMix loss + backward + optimizer step on synthetic data
for a grid of (encoder, img_size, batch_size, amp) and records peak VRAM, step
time and failures. No dataset needed.

Each trial runs in its OWN SUBPROCESS. That is not tidiness: on a display GPU,
a step longer than the Windows TDR timeout (default 2 s) gets the kernel killed
and the CUDA context destroyed, after which every later call in that process
fails with "the launch timed out and was terminated". In-process sweeping loses
every remaining result to the first slow configuration.

Run from the LoMix repo root:
    python ..\..\tools\memfit.py --encoders pvt_v2_b0 pvt_v2_b2 --batch_sizes 1 2 3 4 6
"""
import argparse, json, os, platform, subprocess, sys
from datetime import datetime

TDR_SECONDS = 2.0     # Windows default GPU watchdog; a step above this can be killed


def find_repo_root(start):
    d = os.path.abspath(start)
    while d != os.path.dirname(d):
        if os.path.isfile(os.path.join(d, "lib", "networks.py")):
            return d
        d = os.path.dirname(d)
    return None


# --------------------------------------------------------------- child process

def run_trial(enc, img, bs, amp, num_classes, supervision, steps=3):
    """Executed in the child. Prints one JSON object on stdout."""
    import time
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from lib.networks import EMCADNet
    from utils.utils import DiceLoss

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    model = EMCADNet(num_classes=num_classes, encoder=enc, pretrain=False).to(device)
    model.train()

    if supervision == "lomix":
        operations, learnable = ["add", "mul", "wf", "concat"], True
    elif supervision == "mutation":
        operations, learnable = ["add"], False
    else:
        operations, learnable = [], False

    loss_kind = "repo"
    try:
        from trainer import CombinatorialMutationsLossModule
        loss_module = CombinatorialMutationsLossModule(
            4, num_classes, selecetd_num_maps=4,
            operations=operations, use_learnable_weights=learnable).to(device)
    except Exception as e:
        loss_module, loss_kind = None, "proxy ({})".format(type(e).__name__)

    ce = nn.CrossEntropyLoss()
    dice = DiceLoss(num_classes)
    params = list(model.parameters())
    if loss_module is not None:
        params += list(loss_module.parameters())
    opt = optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    x = torch.randn(bs, 1, img, img, device=device)
    y = torch.randint(0, num_classes, (bs, img, img), device=device)

    times = []
    for i in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            P = model(x, mode="train")
        if amp:
            P = [p.float() for p in P] if isinstance(P, list) else P.float()
        if loss_module is not None and supervision in ("lomix", "mutation", "deep_supervision"):
            loss, _, _ = loss_module(P, y, ce, dice)
        else:
            loss = 0.3 * ce(P[-1], y.long()) + 0.7 * dice(P[-1], y, softmax=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        if i > 0:                      # drop first step (allocator warm-up)
            times.append((time.perf_counter() - t0) * 1000.0)

    step_ms = sum(times) / max(len(times), 1)
    print("__TRIAL__" + json.dumps({
        "status": "fit",
        "peak_train_vram_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "step_ms": round(step_ms, 1),
        "tdr_risk": step_ms > TDR_SECONDS * 1000,
        "loss_module": loss_kind,
    }))


# -------------------------------------------------------------- parent process

def spawn(here, repo, enc, img, bs, amp, num_classes, supervision, timeout):
    cmd = [sys.executable, "-W", "ignore", os.path.join(here, "memfit.py"),
           "--_trial", "--repo", repo, "--encoder", enc,
           "--img_size", str(img), "--batch_size", str(bs),
           "--supervision", supervision, "--num_classes", str(num_classes)]
    if amp:
        cmd.append("--amp_on")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=repo)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT (>{}s)".format(timeout)}

    for line in p.stdout.splitlines():
        if line.startswith("__TRIAL__"):
            return json.loads(line[len("__TRIAL__"):])

    err = (p.stderr or "").strip()
    if "out of memory" in err.lower():
        return {"status": "OOM"}
    if "launch timed out" in err.lower() or "CUDNN_STATUS_INTERNAL_ERROR" in err:
        return {"status": "TDR/driver reset (step exceeded the GPU watchdog)"}
    last = err.splitlines()[-1] if err else "no output, exit {}".format(p.returncode)
    return {"status": "ERROR: " + last[:160]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--encoders", nargs="+", default=["pvt_v2_b0", "pvt_v2_b2"])
    ap.add_argument("--img_sizes", type=int, nargs="+", default=[224])
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--supervision", default="lomix",
                    choices=["lomix", "mutation", "deep_supervision", "last_layer"])
    ap.add_argument("--num_classes", type=int, default=9)
    ap.add_argument("--amp", nargs="+", default=["off", "on"], choices=["off", "on"])
    ap.add_argument("--timeout", type=int, default=300, help="seconds per trial")
    ap.add_argument("--out", default=None)
    # child-mode flags
    ap.add_argument("--_trial", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--encoder", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--img_size", type=int, default=224, help=argparse.SUPPRESS)
    ap.add_argument("--batch_size", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--amp_on", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo = args.repo or find_repo_root(os.getcwd()) or find_repo_root(here)
    if repo is None:
        sys.exit("Could not locate the LoMix repo root. Pass --repo <path to LoMix>.")
    repo = os.path.abspath(repo)
    sys.path.insert(0, repo)
    os.chdir(repo)

    if args._trial:
        run_trial(args.encoder, args.img_size, args.batch_size, args.amp_on,
                  args.num_classes, args.supervision)
        return

    import torch
    if not torch.cuda.is_available():
        sys.exit("No CUDA device visible.")
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    free = torch.cuda.mem_get_info()[0] / 2 ** 20
    env = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "host": platform.node(), "gpu": torch.cuda.get_device_name(0),
           "capability": list(torch.cuda.get_device_capability(0)),
           "vram_total_MiB": round(total), "vram_free_at_start_MiB": round(free),
           "torch": torch.__version__, "supervision": args.supervision,
           "tdr_threshold_s": TDR_SECONDS}
    print(json.dumps(env, indent=2))
    if free < total * 0.9:
        print("\nWARNING: {:.0f} of {:.0f} MiB free -- other processes (browser, chat "
              "apps) hold VRAM. Close them before trusting these limits."
              .format(free, total))
    print("\nEach trial runs in its own subprocess, so a watchdog reset costs one "
          "row, not the sweep.\n")
    del torch

    rows = []
    for enc in args.encoders:
        for img in args.img_sizes:
            for amp_s in args.amp:
                for bs in sorted(args.batch_sizes):
                    tag = "{} img={} bs={} amp={}".format(enc, img, bs, amp_s)
                    r = spawn(here, repo, enc, img, bs, amp_s == "on",
                              args.num_classes, args.supervision, args.timeout)
                    r.update(encoder=enc, img_size=img, batch_size=bs, amp=amp_s)
                    rows.append(r)
                    if r["status"] == "fit":
                        flag = "  <-- exceeds TDR watchdog" if r.get("tdr_risk") else ""
                        print("  OK   {:42s} peak={:8.1f} MiB  step={:8.1f} ms{}".format(
                            tag, r["peak_train_vram_MiB"], r["step_ms"], flag))
                    else:
                        print("  FAIL {:42s} {}".format(tag, r["status"]))
                        break          # larger batches will not do better

    out = args.out or os.path.join(here, "..", "logs",
                                   "memfit_{}_{}.json".format(platform.node(), args.supervision))
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"env": env, "rows": rows}, f, indent=2)

    print("\n| encoder | img | bs | amp | peak MiB | step ms | TDR risk | status |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["encoder"], r["img_size"], r["batch_size"], r["amp"],
            r.get("peak_train_vram_MiB", "-"), r.get("step_ms", "-"),
            "yes" if r.get("tdr_risk") else ("no" if r["status"] == "fit" else "-"),
            r["status"]))
    print("\nSaved -> " + out)


if __name__ == "__main__":
    main()
