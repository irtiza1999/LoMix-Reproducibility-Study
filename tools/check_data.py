r"""
Step 1 -- verify the Synapse data, split lists and PVTv2 weights are in place
BEFORE launching a run that would otherwise die 30 seconds in (or, worse,
train on the wrong label convention for hours).

Run from the LoMix repo root:
    python ..\..\tools\check_data.py --repo .
"""
import argparse, os, sys, glob


def find_repo_root(start):
    d = os.path.abspath(start)
    while d != os.path.dirname(d):
        if os.path.isfile(os.path.join(d, "lib", "networks.py")):
            return d
        d = os.path.dirname(d)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--root_path", default="./data/synapse/train_npz_new")
    ap.add_argument("--volume_path", default="./data/synapse/test_vol_h5_new")
    ap.add_argument("--list_dir", default="./lists/lists_Synapse")
    ap.add_argument("--encoder", default="pvt_v2_b2")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo = args.repo or find_repo_root(os.getcwd()) or find_repo_root(here)
    if repo is None:
        sys.exit("Could not locate the LoMix repo root. Pass --repo <path to LoMix>.")
    repo = os.path.abspath(repo)      # resolve BEFORE chdir, or a relative --repo doubles
    os.chdir(repo)
    print("repo: " + repo + "\n")

    problems = []

    # --- training slices -----------------------------------------------------
    npz = sorted(glob.glob(os.path.join(args.root_path, "*.npz")))
    print("train slices  : {:5d}  in {}".format(len(npz), args.root_path))
    if not npz:
        problems.append("no .npz slices under " + args.root_path)
    elif len(npz) != 2211:
        print("                (paper/TransUNet split is 2211 slices from 18 scans -- "
              "yours differs; note this in the summary)")

    # --- test volumes --------------------------------------------------------
    # "*.npy.h5" is a subset of "*.h5" -- union them, or every volume counts twice
    vol = sorted(set(glob.glob(os.path.join(args.volume_path, "*.h5"))) |
                 set(glob.glob(os.path.join(args.volume_path, "*.npy.h5"))))
    print("test volumes  : {:5d}  in {}".format(len(vol), args.volume_path))
    if not vol:
        problems.append("no test volumes under " + args.volume_path)
    elif len(vol) != 12:
        print("                (TransUNet test split is 12 volumes -- yours differs)")

    # --- split lists ---------------------------------------------------------
    for name in ("train.txt", "test_vol.txt"):
        p = os.path.join(args.list_dir, name)
        if os.path.isfile(p):
            n = sum(1 for line in open(p) if line.strip())
            print("list {:<13s}: {:5d} entries".format(name, n))
        else:
            problems.append("missing list file " + p +
                            "  (copy lists/lists_Synapse/ from EMCAD, CASCADE or TransUNet)")

    # --- pretrained encoder --------------------------------------------------
    w = os.path.join("./pretrained_pth/pvt", args.encoder + ".pth")
    if os.path.isfile(w):
        print("encoder weights: {}  ({:.1f} MB)".format(w, os.path.getsize(w) / 1e6))
    else:
        problems.append("missing " + w + "  (accuracy without it is NOT comparable "
                        "to the paper -- only run with --no_pretrain for timing tests)")

    # --- label convention ----------------------------------------------------
    ds = os.path.join("utils", "dataset_synapse.py")
    if os.path.isfile(ds):
        text = open(ds, encoding="utf-8").read()
        # The remap collapses the 14-label BTCV ground truth to the 9 classes the
        # paper trains on. It is guarded by `if self.nclass == 9:` and only counts
        # when it is not commented out -- the file also contains a dead copy.
        active = [ln for ln in text.splitlines()
                  if "label[label==13]" in ln and not ln.strip().startswith("#")]
        print("\n14->9 label remap ACTIVE in dataset_synapse.py: {}".format(bool(active)))
        print("  -> TransUNet preprocessed archive  : this block MUST be removed"
              "\n     LoMix / EMCAD preprocessed archive: keep it as is"
              "\n     Applying it twice silently zeroes real organ labels."
              "\n     Record which archive you used in the execution summary.")

    print()
    if problems:
        print("BLOCKING PROBLEMS:")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    print("DATA OK")


if __name__ == "__main__":
    main()
