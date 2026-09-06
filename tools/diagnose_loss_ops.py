r"""
Step 1 -- evidence for the two defects found in LoMix's combinatorial loss.

Claim 1: the 'concat' operation builds a NEW randomly-initialised nn.Conv2d on
         every forward pass (trainer.py, `elif op == 'concat'`). It is never
         registered as a submodule and never reaches the optimizer, so the branch
         cannot learn and injects a fresh random projection each step.
Test:    call the loss twice with IDENTICAL inputs under torch.no_grad and
         compare the 'concat' fused maps. Deterministic ops must match exactly;
         a re-initialised conv will not. Also count how many of the loss module's
         parameters the optimizer would actually see.

Claim 2: the 'mul' operation multiplies raw logits, so a k-map combination is a
         k-fold product. Magnitudes grow like |logit|^k and overflow CE/Dice.
Test:    report max |value| of each fused map by operation and combination size,
         from real logits produced by the trained checkpoint.

Run from the LoMix repo root:
    python ..\..\tools\diagnose_loss_ops.py --ckpt <path to best.pth>
"""
import argparse, os, sys

import torch
import torch.nn as nn


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
    ap.add_argument("--ckpt", default=None, help="a trained best.pth (optional)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_classes", type=int, default=9)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo = args.repo or find_repo_root(os.getcwd()) or find_repo_root(here)
    repo = os.path.abspath(repo)
    sys.path.insert(0, repo)
    os.chdir(repo)

    from lib.networks import EMCADNet
    from trainer import CombinatorialMutationsLossModule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ops = ["add", "mul", "wf", "concat"]

    model = EMCADNet(num_classes=args.num_classes, encoder="pvt_v2_b2", pretrain=False).to(device)
    if args.ckpt and os.path.isfile(args.ckpt):
        sd = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(sd)
        finite = all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point())
        print("checkpoint: {}\n  all weights finite: {}\n".format(args.ckpt, finite))
    else:
        print("no checkpoint given -- using an untrained model (magnitudes will be small)\n")
    model.eval()

    lm = CombinatorialMutationsLossModule(4, args.num_classes, selecetd_num_maps=4,
                                          operations=ops, use_learnable_weights=True).to(device)

    # ---- what the optimizer actually sees ---------------------------------
    n_params = sum(p.numel() for p in lm.parameters())
    names = [n for n, _ in lm.named_parameters()]
    print("=== Claim 1: 'concat' conv is not a parameter of the loss module ===")
    print("loss-module parameters visible to the optimizer: {} tensors, {} values"
          .format(len(names), n_params))
    print("any parameter name containing 'conv': {}"
          .format([n for n in names if "conv" in n.lower()] or "NONE"))

    torch.manual_seed(0)
    x = torch.randn(1, 1, args.img_size, args.img_size, device=device)
    with torch.no_grad():
        P = model(x, mode="test")
        a = lm(P)          # label_batch=None -> returns the fused maps
        b = lm(P)          # identical inputs, second call

    combos = lm.combination_indices
    idx, report = 0, {}
    for op in ops:
        for comb in combos[op]:
            report.setdefault(op, []).append((comb, a[idx], b[idx]))
            idx += 1

    print("\nsame input, two forward passes -- max |a-b| per operation:")
    for op in ops:
        diffs = [ (x1 - x2).abs().max().item() for _, x1, x2 in report[op] ]
        verdict = "NON-DETERMINISTIC (re-initialised each call)" if max(diffs) > 1e-6 else "deterministic"
        print("  {:<8s} max diff {:.6e}   {}".format(op, max(diffs), verdict))

    # ---- magnitude growth --------------------------------------------------
    print("\n=== Claim 2: 'mul' magnitude grows with combination size ===")
    print("input logits: max |p| = {:.3f}".format(max(p.abs().max().item() for p in P)))
    print("\n  op       k=2         k=3         k=4")
    for op in ops:
        by_k = {}
        for comb, m, _ in report[op]:
            by_k.setdefault(len(comb), []).append(m.abs().max().item())
        row = "  {:<8s}".format(op)
        for k in (2, 3, 4):
            row += " {:>11.3f}".format(max(by_k[k])) if k in by_k else " {:>11s}".format("-")
        print(row)

    print("\nA k-fold product of logits grows like |logit|^k. CE and Dice on that "
          "saturate, and the gradient overflows -- which is where the epoch-13 NaN "
          "came from.")


if __name__ == "__main__":
    main()
