# LoMix Reproducibility Study

Reproducible, reduced-budget experiments for the LoMix supervision ablation on
Synapse, including low-VRAM adaptations and measured failure analysis.

Repository name: `lomix-reproducibility-study`

This repository contains the experiment harness, the patched LoMix source,
environment setup, measured logs, and report deliverables. Dataset files,
pretrained weights, model checkpoints, caches, and other machine-local outputs
are intentionally excluded from Git. Follow `setup/02_data_and_weights.md` to
prepare those files locally.

## Quick start

```powershell
conda activate lomix
python tools\check_data.py --repo repos\LoMix
python tools\apply_low_vram_patch.py --repo repos\LoMix
.\run_ablation.ps1
```

To continue an interrupted sweep from its saved full-state checkpoint:

```powershell
.\run_ablation.ps1 -Resume -ResumeStamp <stamp> -Supervision lomix
```

See the sections below for the complete protocol and the reduced-budget
disclosure.

# Step 1 — Codebase Exploration & Experimental Verification

Screening submission for Dr. Md Mostafijur Rahman (ECE, Texas Tech).
Paper selected: **LoMix — Learnable Weighted Multi-Scale Logits Mixing for
Medical Image Segmentation** (NeurIPS 2025), code `SLDGroup/LoMix`.

## Why LoMix

Three reasons, in the order they matter:

1. **It is the axis the codebase actually exposes.** `train_synapse_lomix.py`
   takes `--supervision {last_layer, deep_supervision, mutation, lomix}` over a
   single fixed network (`EMCADNet`). That gives a clean four-arm ablation in
   which the network, the data and the schedule are identical and *only the
   supervision differs* — the paper's central claim, isolated, with no
   architecture surgery required.
2. **It is the closest paper to my own work.** LoMix learns *how much to trust
   each prediction scale*. CLEAR-MoE selects layers for expert extraction by a
   calibration criterion; the LiDAR detector (PSC) conditions on its own
   predictions and is evaluated on calibration, not just accuracy. Same
   question, three domains.
3. **Zero inference overhead is separately checkable.** All four arms share one
   inference graph, so the "no inference cost" claim is verified once with a
   direct measurement (Stage A) rather than being taken on trust.

## Hardware reality, stated up front

GTX 960, 4 GB, compute capability 5.2 (Maxwell, no tensor cores).
The paper's setting is batch 6 for 300 epochs. That does not fit here in either
memory or time. The consequences are handled explicitly rather than hidden:

- memory → per-step batch reduced, effective batch restored by **gradient
  accumulation**, plus optional AMP (`tools/apply_low_vram_patch.py`)
- time → **reduced epoch budget**, and validation every *N* epochs instead of
  every epoch (upstream re-runs full-volume inference after each one)

Everything produced here is therefore a **reduced-budget reproduction**, and it
says so on every table. A reduced-budget number compared against a published
300-epoch number without that caveat would be a false claim, not a replication.

## Stages

| Stage | What | Needs data? | Deliverable value |
|---|---|---|---|
| A | Efficiency benchmark: params, GMACs, latency, peak inference VRAM | no | verifies the efficiency/zero-overhead claim directly |
| B | Memory-fit sweep: largest (encoder, img, batch, amp) that trains | no | justifies every config choice with a measurement |
| C | Supervision ablation on Synapse: 4 arms, identical everything else | **yes** | the actual replication result |
| D | Collect logs → tables + learned-weight readout | — | Deliverable 1 and 2 |

Stages A and B run today, before any dataset download finishes. That ordering
is deliberate: it produces real, defensible numbers even if Synapse access or
GPU time turns out to be the bottleneck.

## Commands

```powershell
# 0. environment (once)
powershell -ExecutionPolicy Bypass -File setup\01_create_env.ps1
conda activate lomix

# A. efficiency benchmark (no data needed)
cd repos\LoMix
python ..\..\tools\bench_efficiency.py --encoders pvt_v2_b0 pvt_v2_b2 --img_size 224

# B. what actually fits in 4 GB (no data needed)
python ..\..\tools\memfit.py --encoders pvt_v2_b0 pvt_v2_b2 --batch_sizes 1 2 3 4 6
cd ..\..

# C. data + weights: follow setup\02_data_and_weights.md, then
conda activate lomix
python tools\check_data.py --repo repos\LoMix        # must print DATA OK
python tools\apply_low_vram_patch.py --repo repos\LoMix
.\run_ablation.ps1 -BatchSize 2 -Accum 3 -MaxEpochs 50 -EvalEvery 10

# D. tables
python tools\collect_results.py --stamp <stamp printed by run_ablation>
```

Set `-BatchSize` / `-Accum` from the Stage B table, not from a guess. Keep
`BatchSize * Accum = 6` to match the paper's effective batch.

## Files

```
step1/
├── README.md                     this plan
├── run_ablation.ps1              Stage C sweep, one log + one manifest per run
├── setup/
│   ├── 01_create_env.ps1         conda env `lomix` (timm 0.6.12, torch 2.2.2+cu121)
│   ├── verify_env.py             checks sm_52 coverage, fp16, timm 0.6 APIs
│   └── 02_data_and_weights.md    Synapse / PVTv2 download + the 14→9 class trap
├── tools/
│   ├── bench_efficiency.py       Stage A
│   ├── memfit.py                 Stage B
│   ├── check_data.py             pre-flight: slices, volumes, lists, weights
│   ├── apply_low_vram_patch.py   AMP + accumulation + eval interval (reversible)
│   └── collect_results.py        Stage D
├── repos/                        LoMix, G-CASCADE, EMCAD (upstream, unmodified
│                                 except the documented low-VRAM patch)
├── logs/                         all run logs, manifests, collected JSON
└── deliverables/                 D1 execution summary, D2 report/deck
```

## Ground rules for this submission

The screening email states that blind LLM use is disqualifying. Two rules follow
and are not negotiable:

- **No number appears in any deliverable unless it came out of a run on this
  machine, and the log that produced it is in `logs/`.** Missing results are
  reported as missing.
- Every deviation from the paper's setup — epochs, batch, AMP, validation
  interval, preprocessing archive — is listed in the summary, with the reason.

Discrepancies against the published tables are a *finding*, not a failure, and
are the most useful thing in the report as long as the cause is investigated
(budget? preprocessing archive? label remap? seed?) rather than glossed over.

## Fallback

If the 960 proves too slow for four arms, the honest order of retreat is:

1. drop to `pvt_v2_b0` (all four arms, smaller encoder — still a valid ablation,
   just not at the paper's capacity), then
2. cut arms to `deep_supervision` vs `lomix` (the load-bearing comparison), then
3. move Stage C to Colab/Kaggle T4 with identical commands.

What does not happen: reporting a paper number as a run of ours.
