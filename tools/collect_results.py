r"""
Step 1 -- turn raw run logs into the tables that go into Deliverable 1 and 2.

Parses only what the runs actually printed. Anything the logs do not contain is
reported as null, never filled in from the paper. If an arm crashed, it appears
in the table as a crash, with its exit code.

Usage:
    python tools\collect_results.py                 # every log in step1/logs
    python tools\collect_results.py --stamp 20260904_161200
"""
import argparse, glob, json, os, re
from collections import OrderedDict

RE_PERF = re.compile(r"mean_dice\s*:\s*([0-9.]+),\s*best_dice\s*:\s*([0-9.]+)")
RE_VRAM = re.compile(r"epoch\s+(\d+)\s*:\s*peak_train_vram_MiB\s*:\s*([0-9.]+)")
RE_LOSS = re.compile(r"iteration\s+(\d+),\s*epoch\s+(\d+)\s*:\s*loss\s*:\s*([0-9.eE+-]+),"
                     r"\s*deep_supervision_loss\s*:\s*([0-9.eE+-]+),"
                     r"\s*mutation_loss\s*:\s*([0-9.eE+-]+)")
RE_CFG = re.compile(r"\[low-vram\]\s*amp=(\S+)\s+accum=(\d+)\s+eval_every=(\d+)\s+"
                    r"num_workers=(\S+)\s+per_step_batch=(\d+)\s+effective_batch=(\d+)")
RE_ITERS = re.compile(r"(\d+)\s+iterations per epoch\.\s+(\d+)\s+max iterations")
RE_RUNNER = re.compile(r"\[runner\]\s+exit_code=(-?\d+)\s+wall_clock_min=([0-9.]+)")
RE_ORIG_W = re.compile(r"Original Weights \(softplus\):\s*(.+)")
RE_SYNTH_W = re.compile(r"Synthesized Weights for '([^']+)' \(softplus\):\s*(.+)")
RE_TRAINLEN = re.compile(r"The length of train set is:\s*(\d+)")
RE_OOM = re.compile(r"CUDA out of memory", re.I)
RE_TRACE = re.compile(r"^Traceback \(most recent call last\)", re.M)
RE_NAN = re.compile(r"iteration\s+(\d+),\s*epoch\s+(\d+)\s*:\s*loss\s*:\s*(?:nan|-?inf)",
                    re.I)

# filename: <stamp>_<encoder>_<supervision>_bs<B>x<A>_e<E>_s<S>.log
RE_NAME = re.compile(r"^(?P<stamp>\d{8}_\d{6})_(?P<encoder>.+?)_"
                     r"(?P<supervision>last_layer|deep_supervision|mutation|lomix)_"
                     r"bs(?P<bs>\d+)x(?P<accum>\d+)_e(?P<epochs>\d+)_s(?P<seed>\d+)\.log$")


def floats(s):
    out = []
    for tok in s.replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def read_log(path):
    """Windows PowerShell 5.1's Tee-Object writes UTF-16LE with a BOM and has no
    -Encoding parameter, so run logs are UTF-16 while the trainer's own log.txt
    is UTF-8. Sniff the BOM instead of assuming either."""
    raw = open(path, "rb").read()
    for bom, enc in ((b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"),
                     (b"\xef\xbb\xbf", "utf-8-sig")):
        if raw.startswith(bom):
            return raw.decode(enc, errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def parse(path):
    text = read_log(path)
    name = os.path.basename(path)
    m = RE_NAME.match(name)
    rec = OrderedDict()
    rec["log"] = name
    if m:
        rec.update(stamp=m["stamp"], encoder=m["encoder"], supervision=m["supervision"],
                   per_step_batch=int(m["bs"]), accum=int(m["accum"]),
                   max_epochs=int(m["epochs"]), seed=int(m["seed"]))
    else:
        rec.update(stamp=None, encoder=None, supervision=None)

    cfg = RE_CFG.search(text)
    if cfg:
        rec["amp"] = cfg.group(1)
        rec["eval_every"] = int(cfg.group(3))
        rec["num_workers"] = cfg.group(4)
        rec["effective_batch"] = int(cfg.group(6))

    tl = RE_TRAINLEN.search(text)
    rec["train_slices"] = int(tl.group(1)) if tl else None
    it = RE_ITERS.search(text)
    rec["iters_per_epoch"] = int(it.group(1)) if it else None

    # Divergence: once the loss is NaN the weights are destroyed and every later
    # validation returns 0. Taking max() over the whole run would silently report
    # the pre-divergence peak as if the run had completed.
    nan = RE_NAN.search(text)
    rec["diverged_at_iter"] = int(nan.group(1)) if nan else None
    rec["diverged_at_epoch"] = int(nan.group(2)) if nan else None

    dices = [float(a) for a, _ in RE_PERF.findall(text)]
    rec["n_validations"] = len(dices)
    rec["best_mean_dice"] = round(max(dices), 4) if dices else None
    rec["last_mean_dice"] = round(dices[-1], 4) if dices else None
    rec["dice_curve"] = [round(d, 4) for d in dices]

    vram = [float(v) for _, v in RE_VRAM.findall(text)]
    rec["peak_train_vram_MiB"] = round(max(vram), 1) if vram else None

    losses = RE_LOSS.findall(text)
    if losses:
        rec["first_logged_loss"] = round(float(losses[0][2]), 4)
        rec["last_logged_loss"] = round(float(losses[-1][2]), 4)
        rec["last_epoch_logged"] = int(losses[-1][1])

    run = RE_RUNNER.search(text)
    rec["exit_code"] = int(run.group(1)) if run else None
    rec["wall_clock_min"] = float(run.group(2)) if run else None

    ow = RE_ORIG_W.findall(text)
    rec["final_original_weights"] = floats(ow[-1]) if ow else None
    synth = OrderedDict()
    for op, vals in RE_SYNTH_W.findall(text):
        # print_weights() logs each line twice: once via logging (%-substituted)
        # and once via print(), which passes the % args as separate arguments and
        # so emits the literal "%s". Keep only the substituted logging lines.
        if op == "%s":
            continue
        synth[op] = floats(vals)          # later epochs overwrite earlier ones
    rec["final_synthesized_weights"] = synth or None

    # A resumed arm restarts best_performance at 0 and rebuilds AdamW from
    # scratch, so its "best dice" covers only the epochs after the resume and is
    # NOT comparable with an arm trained straight through. Never let that pass
    # as a clean row.
    res = re.search(r"\[resume\] loaded model from .*?starting epoch (\d+)", text)
    if res is None:
        res = re.search(r"\[resume\] loaded model from", text)
        rec["resumed_from_epoch"] = "unknown" if res else None
    else:
        rec["resumed_from_epoch"] = int(res.group(1))

    fail = []
    if RE_OOM.search(text):
        fail.append("CUDA OOM")
    if RE_TRACE.search(text):
        fail.append("python traceback")
    if rec["exit_code"] not in (0, None):
        fail.append("exit {}".format(rec["exit_code"]))
    rec["failures"] = fail or None
    # Divergence outranks a non-zero exit code: a diverged run is usually killed,
    # and "crashed" would hide the actual cause.
    if rec.get("diverged_at_epoch") is not None:
        rec["status"] = "DIVERGED (NaN) at epoch {}".format(rec["diverged_at_epoch"])
    elif fail:
        rec["status"] = "crashed"
    elif rec.get("resumed_from_epoch") is not None:
        rec["status"] = "RESUMED from epoch {} - not comparable".format(
            rec["resumed_from_epoch"])
    elif dices:
        rec["status"] = "ok"
    else:
        rec["status"] = "no validation recorded"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=None, help="log directory (default step1/logs)")
    ap.add_argument("--stamp", default=None, help="only logs from this sweep")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    logdir = args.logs or os.path.join(here, "..", "logs")
    pattern = os.path.join(logdir, ("{}_*.log".format(args.stamp)) if args.stamp else "*.log")
    files = sorted(f for f in glob.glob(pattern) if not f.endswith("env_setup.log"))
    if not files:
        raise SystemExit("no run logs matched " + pattern)

    recs = [parse(f) for f in files]
    order = {"last_layer": 0, "deep_supervision": 1, "mutation": 2, "lomix": 3}
    recs.sort(key=lambda r: (r.get("stamp") or "", order.get(r.get("supervision"), 9)))

    out = args.out or os.path.join(logdir, "collected_{}.json".format(args.stamp or "all"))
    out = os.path.abspath(out)
    with open(out, "w") as f:
        json.dump(recs, f, indent=2)

    hdr = ["supervision", "encoder", "effective_batch", "max_epochs", "best_mean_dice",
           "peak_train_vram_MiB", "wall_clock_min", "status"]
    print("\n### LoMix supervision ablation (reduced budget)\n")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in recs:
        print("| " + " | ".join(str(r.get(h)) for h in hdr) + " |")

    print("\nReduced-budget reproduction. Epoch counts below the paper's 300 are stated "
          "in the table; do not compare these Dice values to published ones without "
          "repeating that caveat.")

    lom = [r for r in recs if r.get("supervision") == "lomix" and r.get("final_original_weights")]
    if lom:
        r = lom[-1]
        print("\n### Learned mixing weights at the end of the LoMix run (softplus)\n")
        print("per-scale (p4,p3,p2,p1): " +
              " ".join("{:.4f}".format(v) for v in r["final_original_weights"]))
        for op, vals in (r.get("final_synthesized_weights") or {}).items():
            print("{:<8s}: {}".format(op, " ".join("{:.4f}".format(v) for v in vals)))
        print("\nThese are the numbers to reason about in the proposal: which scales and "
              "which mixing operations the model actually up-weights, and whether that "
              "ordering is stable across seeds.")

    print("\nSaved -> " + out)


if __name__ == "__main__":
    main()
