r"""
Step 1 -- adapt the LoMix training loop to a 4 GB Maxwell GPU.

The upstream trainer trains in fp32, with num_workers=8, and runs a full
Synapse volume inference after EVERY epoch. None of that is workable on a
GTX 960 (4 GB, sm_52, no tensor cores). This script applies four minimal,
clearly-scoped edits and records them so they can be reported honestly as
"adaptations", not as changes to the method:

  1. optional AMP (autocast around the encoder/decoder forward only; the
     CE + Dice losses stay in fp32 so the loss maths is unchanged)
  2. optional gradient accumulation, so the *effective* batch size can match
     the paper while the per-step batch fits in VRAM
  3. configurable DataLoader workers (Windows spawn makes 8 workers a
     throughput loss, not a gain)
  4. configurable validation interval + per-epoch peak-VRAM logging

The method itself -- the LoMix loss module, the learnable mixing weights, the
operations set, the network -- is NOT touched.

Usage:
    python apply_low_vram_patch.py --repo ..\repos\LoMix          # apply
    python apply_low_vram_patch.py --repo ..\repos\LoMix --revert # restore
"""
import argparse, os, shutil, sys

TRAINER = "trainer.py"
TRAIN_SCRIPT = "train_synapse_lomix.py"
SUFFIX = ".orig"


def read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)


def sub_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        sys.exit("PATCH FAILED [{}]: expected exactly 1 match, found {}.\n"
                 "The upstream file has changed; re-check the anchor before editing."
                 .format(label, n))
    return text.replace(old, new, 1)


# ---------------------------------------------------------------- trainer.py

CFG_ANCHOR = ("    optimizer = optim.AdamW(list(model.parameters()) + "
              "list(loss_module.parameters()), lr=base_lr, weight_decay=0.0001)")

CFG_NEW = CFG_ANCHOR + """

    # --- low-VRAM adaptation (see tools/apply_low_vram_patch.py) -------------
    _amp = bool(getattr(args, 'amp', False))
    _accum = max(1, int(getattr(args, 'accum', 1)))
    _eval_every = max(1, int(getattr(args, 'eval_every', 1)))
    _scaler = torch.cuda.amp.GradScaler(enabled=_amp)
    _smoke_steps = int(getattr(args, 'smoke_steps', 0))
    logging.info('[low-vram] amp=%s accum=%d eval_every=%d num_workers=%s '
                 'per_step_batch=%d effective_batch=%d'
                 % (_amp, _accum, _eval_every, getattr(args, 'num_workers', 2),
                    batch_size, batch_size * _accum))
    # ------------------------------------------------------------------------"""

# Upstream Windows bug, not an adaptation: worker_init_fn is defined as a local
# function inside trainer_synapse. Windows DataLoader workers start via `spawn`,
# which pickles the init callable, and local functions are not picklable:
#   AttributeError: Can't pickle local object 'trainer_synapse.<locals>.worker_init_fn'
# On Linux `fork` is used and the same code works, which is presumably why it
# shipped. Hoisting the function to module scope and binding the seed with
# functools.partial (which IS picklable) fixes it with identical seeding
# behaviour. num_workers is made configurable at the same time.
SEED_ANCHOR = "from utils.utils import val_single_volume"
SEED_NEW = SEED_ANCHOR + """


def _worker_init_fn(worker_id, base_seed=1234):
    \"\"\"Module-level so Windows `spawn` workers can pickle it (see
    tools/apply_low_vram_patch.py). Seeding is unchanged: base_seed + worker_id.\"\"\"
    random.seed(base_seed + worker_id)"""

WORKERS_OLD = """    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)"""
WORKERS_NEW = """    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=int(getattr(args, 'num_workers', 2)), pin_memory=True,
                             worker_init_fn=functools.partial(_worker_init_fn,
                                                              base_seed=args.seed))"""

FUNCTOOLS_ANCHOR = "from itertools import combinations"
FUNCTOOLS_NEW = "from itertools import combinations\nimport functools"

# Crash-safe checkpointing. A partial run that resumes without the optimizer
# state is NOT a continuation of the same experiment: AdamW's moment estimates
# restart from zero, best_performance resets, and the reported "best dice" then
# covers only the epochs after the restart. That is what invalidated the first
# lomix arm. This saves everything needed to continue identically -- model,
# optimizer, loss-module weights, GradScaler, best_performance, iter_num and
# epoch -- and writes it atomically so a shutdown mid-save cannot corrupt it.
RESUME_OLD = """    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)"""
RESUME_NEW = """    best_performance = 0.0
    _start_epoch = 0
    _state_path = os.path.join(snapshot_path, 'train_state.pth')
    if getattr(args, 'resume', False) and os.path.isfile(_state_path):
        _ck = torch.load(_state_path, map_location=device)
        model.load_state_dict(_ck['model'])
        optimizer.load_state_dict(_ck['optimizer'])
        loss_module.load_state_dict(_ck['loss_module'])
        _scaler.load_state_dict(_ck['scaler'])
        best_performance = _ck['best_performance']
        iter_num = _ck['iter_num']
        _start_epoch = _ck['epoch'] + 1
        logging.info('[resume] full state restored from %s: resuming at epoch %d, '
                     'iter %d, best_dice %.6f (optimizer and scaler included)'
                     % (_state_path, _start_epoch, iter_num, best_performance))
    elif getattr(args, 'resume', False):
        logging.info('[resume] requested but no checkpoint at %s -- starting from scratch'
                     % _state_path)
    iterator = tqdm(range(_start_epoch, max_epoch), ncols=70)

    def _save_train_state(_epoch):
        \"\"\"Everything needed to continue identically. Written atomically
        (tmp + os.replace) so a shutdown mid-write cannot truncate it. Reads
        best_performance/iter_num from the enclosing scope at call time.\"\"\"
        _tmp = _state_path + '.tmp'
        torch.save({'epoch': _epoch,
                    'iter_num': iter_num,
                    'best_performance': best_performance,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss_module': loss_module.state_dict(),
                    'scaler': _scaler.state_dict()}, _tmp)
        os.replace(_tmp, _state_path)"""

SAVE_OLD = """        save_mode_path = os.path.join(snapshot_path, 'last.pth')
        torch.save(model.state_dict(), save_mode_path)"""
SAVE_NEW = """        save_mode_path = os.path.join(snapshot_path, 'last.pth')
        torch.save(model.state_dict(), save_mode_path)
        _save_train_state(epoch_num)"""

LOOP_OLD = "    for epoch_num in iterator:"
LOOP_NEW = """    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    for epoch_num in iterator:"""

FWD_OLD = "            P = model(image_batch, mode='train')"
FWD_NEW = """            with torch.cuda.amp.autocast(enabled=_amp):
                P = model(image_batch, mode='train')
            if _amp:   # losses are computed in fp32 so the loss maths is unchanged
                P = [p.float() for p in P] if isinstance(P, list) else P.float()"""

STEP_OLD = """            optimizer.zero_grad()
            loss.backward()
            optimizer.step()"""
STEP_NEW = """            _scaler.scale(loss / _accum).backward()
            if (i_batch + 1) % _accum == 0:
                _scaler.step(optimizer)
                _scaler.update()
                optimizer.zero_grad(set_to_none=True)"""

# A cheap end-to-end check of the REAL entry point. Two upstream bugs (the dead
# PVT_CASCADE import, and the unpicklable worker_init_fn) were both invisible to
# a test harness that imported EMCADNet directly and built its own DataLoader.
# --smoke_steps runs the actual training script for N iterations and stops, so
# the thing being verified is the thing that will run for 18 hours.
SMOKE_OLD = """            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)"""
SMOKE_NEW = """            iter_num = iter_num + 1
            if _smoke_steps and iter_num >= _smoke_steps:
                # Save on the way out so the smoke path exercises the real
                # checkpoint-writing code, not a separate shortcut.
                _save_train_state(epoch_num)
                logging.info('[smoke] reached %d iterations, loss %f -- state saved '
                             'to %s, stopping before validation'
                             % (iter_num, float(loss), _state_path))
                iterator.close()
                writer.close()
                return "Smoke run finished"
            writer.add_scalar('info/lr', lr_, iter_num)"""

VRAM_OLD = "        # Log the weights.\n        loss_module.print_weights()"
VRAM_NEW = """        logging.info('epoch %d : peak_train_vram_MiB : %.1f'
                     % (epoch_num, torch.cuda.max_memory_allocated() / 2 ** 20))
        torch.cuda.reset_peak_memory_stats()

        # Log the weights.
        loss_module.print_weights()"""

EVAL_OLD = "        performance = inference(args, model, best_performance)"
EVAL_NEW = """        if (epoch_num + 1) % _eval_every == 0 or epoch_num >= max_epoch - 1:
            performance = inference(args, model, best_performance)
        else:
            performance = -1.0   # skipped this epoch; never beats best_performance

        # Upstream calls model.train() ONCE before the epoch loop, while
        # inference() calls model.eval() at the end of every epoch and never
        # switches back -- so from epoch 1 onward training runs in eval mode
        # (decoder BatchNorm uses frozen running stats, dropout/drop-path off).
        # Default OFF: the published numbers were produced with this behaviour,
        # so leaving it intact is what "replication" means. Turn it on only as
        # an explicit, separately reported experiment.
        if getattr(args, 'restore_train_mode', False):
            model.train()"""

# ------------------------------------------------------- train_synapse_lomix.py

# Upstream bug, not an adaptation: train_synapse_lomix.py imports PVT_CASCADE,
# which is defined nowhere in the repo (lib/networks.py defines only EMCADNet).
# The only other reference is a commented-out model line. As shipped, the
# released training script therefore fails at import with
#   ImportError: cannot import name 'PVT_CASCADE' from 'lib.networks'
# before it can parse a single argument. Dropping the dead name is the minimal
# fix and changes no behaviour -- EMCADNet is what the script instantiates.
IMPORT_OLD = "from lib.networks import PVT_CASCADE, EMCADNet"
IMPORT_NEW = ("from lib.networks import EMCADNet  # PVT_CASCADE removed: "
              "not defined anywhere in this repo (see tools/apply_low_vram_patch.py)")

ARGS_OLD = "args = parser.parse_args()"
ARGS_NEW = """# --- low-VRAM adaptation flags (see tools/apply_low_vram_patch.py) ---
parser.add_argument('--amp', action='store_true', default=False,
                    help='mixed-precision forward (fp32 losses); cuts activation memory')
parser.add_argument('--accum', type=int, default=1,
                    help='gradient accumulation steps; effective batch = batch_size * accum')
parser.add_argument('--num_workers', type=int, default=2,
                    help='DataLoader workers (8 is a bad default on Windows)')
parser.add_argument('--eval_every', type=int, default=1,
                    help='run full-volume validation every N epochs')
parser.add_argument('--resume', action='store_true', default=False,
                    help='continue from train_state.pth in the snapshot dir if present, '
                         'restoring optimizer/scaler/best_dice/iter so the run is '
                         'identical to an uninterrupted one')
parser.add_argument('--smoke_steps', type=int, default=0,
                    help='stop after N training iterations and skip validation. '
                         'Verification only -- exercises the real entry point cheaply.')
parser.add_argument('--restore_train_mode', action='store_true', default=False,
                    help='call model.train() after each validation. Upstream does not, '
                         'so after epoch 0 training continues in eval mode. Default OFF '
                         'preserves upstream behaviour; enable only as a reported experiment.')

args = parser.parse_args()"""


def apply(repo):
    tp, sp = os.path.join(repo, TRAINER), os.path.join(repo, TRAIN_SCRIPT)
    for p in (tp, sp):
        if not os.path.isfile(p):
            sys.exit("Not found: " + p)
        if not os.path.isfile(p + SUFFIX):
            shutil.copy2(p, p + SUFFIX)
            print("backup -> " + os.path.basename(p) + SUFFIX)

    t = read(tp + SUFFIX)                      # always patch from the pristine copy
    t = sub_once(t, FUNCTOOLS_ANCHOR, FUNCTOOLS_NEW, "functools import")
    t = sub_once(t, SEED_ANCHOR, SEED_NEW, "module-level worker_init_fn")
    t = sub_once(t, WORKERS_OLD, WORKERS_NEW, "dataloader workers")
    t = sub_once(t, CFG_ANCHOR, CFG_NEW, "amp/accum config")
    t = sub_once(t, RESUME_OLD, RESUME_NEW, "full-state resume")
    t = sub_once(t, SAVE_OLD, SAVE_NEW, "atomic checkpoint save")
    t = sub_once(t, LOOP_OLD, LOOP_NEW, "epoch loop preamble")
    t = sub_once(t, FWD_OLD, FWD_NEW, "autocast forward")
    t = sub_once(t, STEP_OLD, STEP_NEW, "scaled backward + accumulation")
    t = sub_once(t, SMOKE_OLD, SMOKE_NEW, "smoke-step early exit")
    t = sub_once(t, VRAM_OLD, VRAM_NEW, "peak vram logging")
    t = sub_once(t, EVAL_OLD, EVAL_NEW, "validation interval")
    write(tp, t)
    print("patched " + TRAINER)

    s = read(sp + SUFFIX)
    s = sub_once(s, IMPORT_OLD, IMPORT_NEW, "dead PVT_CASCADE import")
    s = sub_once(s, ARGS_OLD, ARGS_NEW, "cli flags")
    write(sp, s)
    print("patched " + TRAIN_SCRIPT)

    import py_compile
    for p in (tp, sp):
        py_compile.compile(p, doraise=True)
    print("both files compile cleanly.")

    # Compiling is not enough: the failure this catches is an ImportError at
    # module load, which only shows up when the entry point is actually run.
    import subprocess
    r = subprocess.run([sys.executable, "-W", "ignore", TRAIN_SCRIPT, "--help"],
                       cwd=repo, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()
        print("\nENTRY POINT STILL BROKEN:")
        for line in tail[-6:]:
            print("  " + line)
        sys.exit(1)
    print("entry point imports and parses arguments cleanly.")
    print("\nNew flags: --amp --accum N --num_workers N --eval_every N")


def revert(repo):
    for name in (TRAINER, TRAIN_SCRIPT):
        p = os.path.join(repo, name)
        if os.path.isfile(p + SUFFIX):
            shutil.copy2(p + SUFFIX, p)
            print("reverted " + name)
        else:
            print("no backup for " + name + "; nothing to revert")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to the LoMix repo root")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    revert(repo) if a.revert else apply(repo)
