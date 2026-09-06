# Step 1 environment for the LoMix / EMCAD / G-CASCADE replication.
#
# Why a dedicated env and not the base anaconda env:
#   - the repos need timm 0.6.x APIs (timm.models.layers.trunc_normal_tf_,
#     timm.models.helpers.named_apply). Base has timm 1.0.26, which moved them.
#   - the repos also pin numpy 1.22 / albumentations 1.1; base must stay usable.
#
# Why NOT the torch version in the repo README (1.11.0+cu113):
#   - this machine is a GTX 960 (Maxwell, compute capability 5.2). torch
#     2.2.2+cu121 ships sm_50 kernels, which run on sm_52, and it is already
#     verified working here. Old cu113 wheels add risk for no benefit.
#
# Idempotent: safe to re-run. Run from an "Anaconda PowerShell Prompt".
#
# Note on style: native commands are called directly and checked via
# $LASTEXITCODE. A helper function taking an `$Args` parameter would silently
# collide with PowerShell's automatic `$args` variable and run python with no
# arguments at all -- which is exactly how this script failed the first time.

$ErrorActionPreference = "Continue"   # native stderr must not abort the script
$EnvName = "lomix"

function Assert-LastOk([string]$What) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "FAILED: $What (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}

# --- locate or create the env ------------------------------------------------
$Prefix = Join-Path $env:USERPROFILE "anaconda3\envs\$EnvName"
$Py = Join-Path $Prefix "python.exe"

if (-not (Test-Path $Py)) {
    Write-Host "==> creating conda env '$EnvName' (python 3.10)"
    conda create -n $EnvName python=3.10 -y
    Assert-LastOk "conda create"
}
if (-not (Test-Path $Py)) {
    Write-Host "FAILED: env python not found at $Py" -ForegroundColor Red
    exit 1
}
Write-Host "==> env python: $Py"

# --- dependencies ------------------------------------------------------------
#
# ORDER MATTERS, and getting it wrong is silent.
#
# timm / thop / ptflops / torchprofile all declare an unpinned `torch`
# dependency. Installing them AFTER a pinned torch lets pip re-resolve and
# replace torch 2.2.2+cu121 with the current PyPI default (a +cpu build) while
# leaving the cu121 CUDA DLLs on disk. The result is two torch dist-infos, a
# mixed DLL tree, and `OSError: [WinError 127] ... c10_cuda.dll` at import --
# which is exactly how this script failed the second time.
#
# So: plain deps first, pinned torch second, torch-dependent tools last with
# --no-deps so they cannot touch torch at all. numpy is re-pinned at the end
# because the torch wheel pulls numpy 2.x.

Write-Host "==> upgrading pip"
& $Py -m pip install --upgrade pip --quiet
Assert-LastOk "pip upgrade"

Write-Host "==> [1/4] removing any previous torch (two dist-infos = broken tree)"
& $Py -m pip uninstall -y torch torchvision 2>&1 | Out-Null
& $Py -m pip uninstall -y torch torchvision 2>&1 | Out-Null   # second pass: stacked installs
$TorchDir = Join-Path $Prefix "Lib\site-packages\torch"
if (Test-Path $TorchDir) {
    Write-Host "    removing leftover $TorchDir"
    Remove-Item -Recurse -Force $TorchDir
}

Write-Host "==> [2/4] installing torch-independent dependencies"
& $Py -m pip install "albumentations==1.1.0" "segmentation-mask-overlay==0.3.4" "huggingface-hub<1.0" medpy SimpleITK nibabel h5py scipy pandas matplotlib seaborn scikit-image scikit-learn opencv-python tqdm einops tensorboardX pyyaml tabulate
Assert-LastOk "base dependency install"

Write-Host "==> [3/4] installing torch 2.2.2+cu121 / torchvision 0.17.2+cu121 (sm_50 kernels)"
& $Py -m pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
Assert-LastOk "torch install"

Write-Host "==> [4/4] installing torch-dependent tools with --no-deps, then re-pinning numpy"
& $Py -m pip install --no-deps "timm==0.6.12" thop ptflops torchprofile
Assert-LastOk "torch-dependent tool install"
& $Py -m pip install "numpy<2"
Assert-LastOk "numpy pin"

# --- verification ------------------------------------------------------------
Write-Host "==> verifying"
$Verify = Join-Path $PSScriptRoot "verify_env.py"
& $Py -W ignore $Verify
Assert-LastOk "environment verification"

Write-Host ""
Write-Host "Done. Activate with:  conda activate $EnvName"
