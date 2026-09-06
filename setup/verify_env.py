"""Verify the `lomix` env can actually run the LoMix code paths on this GPU.

Checked here rather than in the shell script so failures are loud and specific.
"""
import sys

ok = True


def check(label, fn):
    global ok
    try:
        print("  {:<34s} {}".format(label, fn()))
    except Exception as e:
        ok = False
        print("  {:<34s} FAILED: {}: {}".format(label, type(e).__name__, e))


def torch_info():
    import torch
    if torch.version.cuda is None:
        raise RuntimeError(
            "this is a CPU-only torch build ({}). A later pip install re-resolved "
            "the pinned cu121 wheel -- reinstall with --no-deps ordering.".format(
                torch.__version__))
    return "{} (cuda {})".format(torch.__version__, torch.version.cuda)


def torch_tree_clean():
    """Two torch dist-infos means a stacked install and a mixed DLL tree."""
    import glob, os, torch
    sp = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
    dists = [os.path.basename(p) for p in glob.glob(os.path.join(sp, "torch-*.dist-info"))]
    if len(dists) > 1:
        raise RuntimeError("multiple torch dist-infos present: {}".format(dists))
    return dists[0] if dists else "no dist-info found"


def gpu_info():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    cap = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    sm = "sm_{}{}".format(*cap)
    # Maxwell sm_52 runs sm_50 cubins; check the major version is covered.
    covered = any(a.startswith("sm_{}".format(cap[0])) for a in archs)
    if not covered:
        raise RuntimeError("{} not covered by build archs {}".format(sm, archs))
    return "{} {} (build archs cover sm_{}x)".format(
        torch.cuda.get_device_name(0), sm, cap[0])


def gpu_math():
    import torch
    a = torch.randn(1024, 1024, device="cuda")
    b = (a @ a).float().mean().item()
    h = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    c = (h @ h).float().mean().item()
    return "fp32 ok ({:.4f}), fp16 ok ({:.4f})".format(b, c)


def timm_apis():
    import timm
    from timm.models.layers import trunc_normal_tf_, DropPath, to_2tuple, trunc_normal_  # noqa: F401
    from timm.models.helpers import named_apply  # noqa: F401
    from timm.models.registry import register_model  # noqa: F401
    return "{} (0.6.x APIs present)".format(timm.__version__)


def numpy_ver():
    import numpy
    if numpy.__version__.startswith("2."):
        raise RuntimeError("numpy 2.x breaks the pinned albumentations/medpy stack")
    return numpy.__version__


def repo_deps():
    import medpy.metric, SimpleITK, ptflops, thop, tensorboardX, h5py, cv2  # noqa: F401
    import scipy, pandas, matplotlib, seaborn, albumentations  # noqa: F401
    from segmentation_mask_overlay import overlay_masks  # noqa: F401
    return "all present"


print("python  ", sys.version.split()[0])
check("torch", torch_info)
check("torch install tree", torch_tree_clean)
check("gpu", gpu_info)
check("gpu math", gpu_math)
check("timm", timm_apis)
check("numpy", numpy_ver)
check("repo deps", repo_deps)

print()
if ok:
    print("ENVIRONMENT OK")
else:
    print("ENVIRONMENT INCOMPLETE -- fix the FAILED lines above before running experiments")
    sys.exit(1)
