r"""
Step 1 -- prove the whole path works on REAL data before committing ~20 hours.

Exercises exactly what a real run exercises: the Synapse dataloader and its
augmentation, the patched training step (accumulation + optional AMP), the
LoMix loss module, backward, and the optimizer -- plus one pass of the
validation path on a single test volume, which is where shape and label-range
mistakes usually surface.

Run from the LoMix repo root:
    python ..\..\tools\smoke_test.py --iters 30
"""
import argparse, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


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
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_classes", type=int, default=9)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo = args.repo or find_repo_root(os.getcwd()) or find_repo_root(here)
    if repo is None:
        sys.exit("Could not locate the LoMix repo root. Pass --repo <path to LoMix>.")
    repo = os.path.abspath(repo)
    sys.path.insert(0, repo)
    os.chdir(repo)

    from lib.networks import EMCADNet
    from utils.dataset_synapse import Synapse_dataset, RandomGenerator
    from utils.utils import DiceLoss
    from trainer import CombinatorialMutationsLossModule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: {}\n".format(device))

    # ---- 1. training data --------------------------------------------------
    db = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                         nclass=args.num_classes,
                         transform=transforms.Compose(
                             [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("[1] train set        : {} slices".format(len(db)))
    loader = DataLoader(db, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    s = db[0]
    img, lab = s["image"], s["label"]
    print("    sample image     : {} {} range [{:.3f}, {:.3f}]".format(
        tuple(img.shape), img.dtype, float(img.min()), float(img.max())))
    print("    sample label     : {} {} values {}".format(
        tuple(lab.shape), lab.dtype, sorted(set(np.asarray(lab).ravel().tolist()))))

    # ---- 2. label range ----------------------------------------------------
    seen = set()
    for i in range(min(200, len(db))):
        seen |= set(np.asarray(db[i]["label"]).ravel().tolist())
    lo, hi = int(min(seen)), int(max(seen))
    print("[2] labels over 200 slices: {} .. {} ({} distinct)".format(lo, hi, len(seen)))
    if hi > args.num_classes - 1:
        sys.exit("FAIL: label {} exceeds num_classes-1 ({}). The 14->9 remap in "
                 "utils/dataset_synapse.py is wrong for this archive.".format(
                     hi, args.num_classes - 1))
    print("    consistent with num_classes={} -- remap setting is correct for this archive"
          .format(args.num_classes))

    # ---- 3. model + LoMix loss --------------------------------------------
    model = EMCADNet(num_classes=args.num_classes, encoder=args.encoder, pretrain=True).to(device)
    model.train()
    loss_module = CombinatorialMutationsLossModule(
        4, args.num_classes, selecetd_num_maps=4,
        operations=["add", "mul", "wf", "concat"], use_learnable_weights=True).to(device)
    ce, dice = nn.CrossEntropyLoss(), DiceLoss(args.num_classes)
    opt = optim.AdamW(list(model.parameters()) + list(loss_module.parameters()),
                      lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    print("[3] model + LoMix loss built (pretrained encoder loaded)")

    # ---- 4. training steps -------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    losses, t0, n = [], time.time(), 0
    opt.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        if n >= args.iters:
            break
        x = batch["image"].to(device)
        y = batch["label"].squeeze(1).to(device)
        with torch.cuda.amp.autocast(enabled=args.amp):
            P = model(x, mode="train")
        if args.amp:
            P = [p.float() for p in P]
        loss, ds, mut = loss_module(P, y, ce, dice)
        scaler.scale(loss / args.accum).backward()
        if (i + 1) % args.accum == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
        n += 1
        if n in (1, args.iters // 2, args.iters):
            print("    iter {:3d}  loss {:.4f}  (deep_sup {:.4f}, mutation {:.4f})".format(
                n, float(loss), float(ds), float(mut)))
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0
    print("[4] {} steps in {:.1f}s  ({:.0f} ms/step)  peak train VRAM {:.1f} MiB".format(
        n, dt, dt * 1000 / max(n, 1), peak))
    print("    loss {:.4f} -> {:.4f} (mean of first 5 vs last 5)".format(
        float(np.mean(losses[:5])), float(np.mean(losses[-5:]))))

    # ---- 5. validation path on one volume ---------------------------------
    from utils.utils import val_single_volume
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol",
                              list_dir=args.list_dir, nclass=args.num_classes)
    print("[5] test set         : {} volumes".format(len(db_test)))
    tl = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    sample = next(iter(tl))
    model.eval()
    t1 = time.time()
    metric = val_single_volume(sample["image"], sample["label"], model,
                               classes=args.num_classes,
                               patch_size=[args.img_size, args.img_size],
                               case=sample["case_name"][0], z_spacing=1)
    print("    volume {} shape {} -> per-class dice {}".format(
        sample["case_name"][0], tuple(sample["image"].shape[1:]),
        [round(float(m), 4) for m in metric]))
    print("    validation of one volume took {:.1f}s -> full 12-volume pass "
          "~{:.1f} min".format(time.time() - t1, (time.time() - t1) * 12 / 60))

    print("\nSMOKE TEST PASSED -- the full path runs on real data.")
    print("Dice above is from an untrained decoder and means nothing yet.")


if __name__ == "__main__":
    main()
