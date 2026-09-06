# Step 1 / Stage C -- the LoMix supervision ablation on Synapse.
#
# This is the experiment that actually answers a question: LoMix's own claim is
# that *learnable weighted mixing of multi-scale logits* beats plain deep
# supervision and beats unweighted mutation supervision, at zero inference cost.
# The repo exposes exactly that axis as --supervision, so all four arms share
# one network, one dataset, one schedule and differ only in the loss.
#
#   last_layer        final logit only            (no deep supervision)
#   deep_supervision  per-scale losses, uniform   (classic DS baseline)
#   mutation          + unweighted logit mixing   (ablates the LEARNED weights)
#   lomix             + learnable mixing weights  (the paper's method)
#
# Because all four arms use the identical backbone and identical inference
# graph, the inference-cost claim is verified separately and once, by
# tools/bench_efficiency.py -- not per arm.
#
# BEFORE RUNNING:
#   1. conda activate lomix
#   2. python tools\check_data.py --repo repos\LoMix      (must print DATA OK)
#   3. python tools\memfit.py --repo repos\LoMix          (pick BatchSize/Amp below)
#   4. python tools\apply_low_vram_patch.py --repo repos\LoMix
#
# Defaults below are set from MEASUREMENTS, not guesses (logs/memfit_*.json,
# GTX 960 4GB, 224px, effective batch 6):
#
#   arm               peak VRAM   step      vs last_layer
#   last_layer         1008 MiB    649 ms      1.00x
#   deep_supervision   1054 MiB    708 ms      1.09x
#   mutation           1268 MiB    938 ms      1.45x
#   lomix              2290 MiB   1649 ms      2.54x
#
# So LoMix costs ~2.5x step time and ~2.3x training memory over a single-logit
# baseline, while adding nothing at inference. AMP is OFF by default: on a card
# with no tensor cores it measured 2-4x SLOWER for ~5% memory saved.
#
# Step timing here is contention-dependent: the same config measured 1432 ms and
# 2193 ms minutes apart (identical peak memory). Batch 3 therefore sits on the
# wrong side of the 2 s watchdog on a busy desktop, so the default is batch 2
# with accumulation 3. Expect roughly 20-25 min/epoch for the lomix arm
# (1106 iters/epoch) and well under half that for last_layer; 20 epochs across
# all four arms is on the order of 18-24 hours. Close GPU-using apps first.
#
# Confirmed on real data by tools/smoke_test.py (bs=2, fp32, lomix arm):
# 1171 ms/step, 1744 MiB peak, loss 45.3 -> 35.2 over 30 steps. At 1106
# iters/epoch that is ~21.6 min/epoch, so the lomix arm alone is ~7.2 h at 20
# epochs. A full 12-volume validation pass is only ~1.5 min, which is why
# EvalEvery is 2 rather than 5.
#
# Reduced-budget disclosure: MaxEpochs is far short of the paper's 300. Every
# number produced here is a *reduced-budget reproduction*, and the execution
# summary must say so in those words. Do not compare a 20-epoch Dice against a
# published 300-epoch Dice without stating the gap.

param(
    [string]   $Encoder    = "pvt_v2_b2",
    [int]      $ImgSize    = 224,
    [int]      $BatchSize  = 2,       # measured: 1633 MiB peak; batch 3 crossed the TDR watchdog on a busy desktop
    [int]      $Accum      = 3,       # effective batch = 2 x 3 = 6, the paper's setting
    [switch]   $Amp,                  # OPT-IN: measured 2-4x SLOWER on this Maxwell card
    [int]      $MaxEpochs  = 20,
    [int]      $EvalEvery  = 2,      # measured: a full 12-volume validation pass is ~1.5 min
    [int]      $NumWorkers = 2,
    [int]      $Seed       = 2222,
    [switch]   $RestoreTrainMode,   # upstream leaves the model in eval() after epoch 0;
                                    # off by default so replication stays faithful
    # Resume is now a single switch. The trainer checkpoints full state
    # (model + optimizer + scaler + best_dice + iter + epoch) after every epoch
    # and restores all of it, so a machine shutdown no longer invalidates a run.
    # The earlier -ResumeStamp/-ResumeFromEpoch pair restored weights ONLY: AdamW
    # restarted from zero and best_dice reset, which is why the first lomix arm
    # had to be discarded. Do not reintroduce it.
    [switch]   $Resume,
    [string]   $ResumeStamp = "",   # reuse an existing sweep stamp (log/manifest naming)

    [string[]] $Supervision = @("last_layer", "deep_supervision", "mutation", "lomix"),
    [string]   $RootPath   = "./data/synapse/train_npz_new",
    [string]   $VolumePath = "./data/synapse/test_vol_h5_new"
)

$ErrorActionPreference = "Continue"

$Step1 = $PSScriptRoot
$Repo  = Join-Path $Step1 "repos\LoMix"
$Logs  = Join-Path $Step1 "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$Py = Join-Path $env:USERPROFILE "anaconda3\envs\lomix\python.exe"
if (-not (Test-Path $Py)) { Write-Host "lomix env python not found at $Py" -ForegroundColor Red; exit 1 }

if (-not (Test-Path (Join-Path $Repo "trainer.py.orig"))) {
    Write-Host "trainer.py.orig not found -- run tools\apply_low_vram_patch.py first." -ForegroundColor Red
    exit 1
}

# Fail fast, through the REAL entry point. Two upstream bugs (a dead
# PVT_CASCADE import, and a worker_init_fn that Windows `spawn` cannot pickle)
# each got discovered four times in a row -- once per arm -- because nothing
# exercised train_synapse_lomix.py itself before the sweep started. A --help
# check would have caught the first and missed the second; only running actual
# training iterations catches both.
Write-Host "preflight: 10 real training steps through train_synapse_lomix.py ..." -NoNewline
Push-Location $Repo
& $Py -W ignore train_synapse_lomix.py --root_path $RootPath --volume_path $VolumePath `
    --encoder $Encoder --supervision lomix --img_size $ImgSize --batch_size $BatchSize `
    --accum $Accum --num_workers $NumWorkers --max_epochs 1 --smoke_steps 10 *> $null
$PreflightCode = $LASTEXITCODE
Pop-Location
if ($PreflightCode -ne 0) {
    Write-Host ""
    Write-Host "PREFLIGHT FAILED (exit $PreflightCode) -- not starting an 18-hour sweep." -ForegroundColor Red
    Write-Host "Reproduce it with:" -ForegroundColor Red
    Write-Host "  cd repos\LoMix; python train_synapse_lomix.py --smoke_steps 10" -ForegroundColor Red
    Write-Host "If the patch is missing or stale: python tools\apply_low_vram_patch.py --repo repos\LoMix" -ForegroundColor Red
    exit 1
}
Write-Host " OK" -ForegroundColor Green

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$IsResume = -not [string]::IsNullOrWhiteSpace($ResumeStamp)
if ($IsResume) { $Stamp = $ResumeStamp }
if ($Resume) { Write-Host "resume: enabled (full state incl. optimizer)" -ForegroundColor Yellow }
$Commit = (git -C $Repo rev-parse --short HEAD 2>$null)
# AMP is opt-in: Stage B measured it 2-4x slower on sm_52 (no tensor cores)
$UseAmp = [bool]$Amp

# One provenance record per sweep, so the summary never has to guess.
$Manifest = [ordered]@{
    stamp = $Stamp; host = $env:COMPUTERNAME; repo_commit = $Commit
    encoder = $Encoder; img_size = $ImgSize
    per_step_batch = $BatchSize; accum = $Accum; effective_batch = $BatchSize * $Accum
    amp = $UseAmp; max_epochs = $MaxEpochs; eval_every = $EvalEvery
    num_workers = $NumWorkers; seed = $Seed; arms = $Supervision
    restore_train_mode = [bool]$RestoreTrainMode; resume = [bool]$Resume
    paper_epochs = 300; paper_batch = 6
    note = "reduced-budget reproduction on a GTX 960 4GB (sm_52); not a full-budget replication"
}
$ManifestPath = Join-Path $Logs "ablation_${Stamp}_manifest.json"
if (-not $IsResume) {
    $Manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 $ManifestPath
}
Write-Host "manifest -> $ManifestPath`n"

Push-Location $Repo
foreach ($sup in $Supervision) {
    $Tag = "${Stamp}_${Encoder}_${sup}_bs${BatchSize}x${Accum}_e${MaxEpochs}_s${Seed}"
    $Log = Join-Path $Logs "$Tag.log"
    if ($IsResume -and $sup -ne "lomix") {
        $existing = Join-Path $Logs "$Tag.log"
        if (Test-Path $existing) {
            $completed = Select-String -Path $existing -Pattern '\[runner\] exit_code=0' -Quiet
            if ($completed) {
                Write-Host "===== $sup already completed; skipping =====" -ForegroundColor DarkGray
                continue
            }
        }
    }
    Write-Host "===== $sup =====" -ForegroundColor Cyan
    Write-Host "log -> $Log"

    $cli = @(
        "-W", "ignore", "train_synapse_lomix.py",
        "--root_path", $RootPath, "--volume_path", $VolumePath,
        "--encoder", $Encoder, "--supervision", $sup,
        "--img_size", $ImgSize, "--batch_size", $BatchSize,
        "--max_epochs", $MaxEpochs, "--seed", $Seed,
        "--accum", $Accum, "--num_workers", $NumWorkers, "--eval_every", $EvalEvery
    )
    if ($Resume) { $cli += "--resume" }
    if ($UseAmp) { $cli += "--amp" }
    if ($RestoreTrainMode) { $cli += "--restore_train_mode" }

    $t0 = Get-Date
    & $Py @cli 2>&1 | Tee-Object -FilePath $Log
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)

    Add-Content -Encoding utf8 $Log "`n[runner] exit_code=$code wall_clock_min=$mins"
    if ($code -ne 0) {
        Write-Host "$sup FAILED (exit $code) after $mins min -- see $Log" -ForegroundColor Red
    } else {
        Write-Host "$sup done in $mins min" -ForegroundColor Green
    }
}
Pop-Location

Write-Host ""
Write-Host "Collect the results with:"
Write-Host "  python tools\collect_results.py --stamp $Stamp"
