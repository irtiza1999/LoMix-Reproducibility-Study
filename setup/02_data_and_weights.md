# Step 1 — data and pretrained weights

Everything below is a **manual download** (Synapse requires a signed data-use
agreement; the mirrors are Google Drive links that need a browser). Nothing in
this pipeline downloads them for you.

Target layout, relative to `step1/repos/LoMix/`:

```
LoMix/
├── data/
│   └── synapse/
│       ├── train_npz_new/      # 2211 .npz slices from the 18 training scans
│       └── test_vol_h5_new/    # 12 .npy.h5 test volumes
├── lists/
│   └── lists_Synapse/          # train.txt / test_vol.txt  (see note 3)
└── pretrained_pth/
    └── pvt/
        ├── pvt_v2_b0.pth
        └── pvt_v2_b2.pth
```

---

## 1. Synapse Multi-organ (BTCV) — required for the ablation

Two routes; **route B is what the repo README recommends and is much faster.**

**Route A — from source.** Register at
<https://www.synapse.org/#!Synapse:syn3193805/wiki/89480>, accept the data-use
agreement, download `RawData`, split into `TrainSet` (18 scans) / `TestSet`
(12 scans) using the TransUNet lists, place under
`./data/synapse/Abdomen/RawData/`, then run
`python ./utils/preprocess_synapse_data.py`.

**Route B — preprocessed mirror.** The LoMix README links the preprocessed
archive here: <https://drive.google.com/file/d/1wvmw8DVyDKr5sOAFn5zUpfhbK4Vxjze4/view>
Unpack into `./data/synapse/`.

> If you instead use the **TransUNet** preprocessed folder
> (<https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd>),
> the README says to delete lines 88–94 of `utils/dataset_synapse.py` — that
> block remaps 14-class ground truth down to 9 classes and would double-apply.
> **Record which archive you used in the execution summary**; it changes the
> ground-truth handling and therefore the numbers.

Sanity check after unpacking (run inside the `lomix` env):

```bash
python ../../tools/check_data.py --repo .
```

## 2. PVTv2 ImageNet weights — required for any accuracy claim

<https://drive.google.com/drive/folders/1d5F1VjEF1AtTkNO93JwVBBSivE8zImiF>
(or the PVT release page: <https://github.com/whai362/PVT/releases/tag/v2>)

Put `pvt_v2_b0.pth` and `pvt_v2_b2.pth` in `./pretrained_pth/pvt/`.

Training with `--no_pretrain` will run, but the resulting Dice is **not**
comparable to the paper and must never be presented as a replication of it.

## 3. Class-name / split lists

`--list_dir ./lists/lists_Synapse` is the default and the folder is **not** in
the GitHub repo. The same lists ship with EMCAD, CASCADE and TransUNet — copy
`lists/lists_Synapse/` from whichever you have. If it is missing, the dataloader
fails immediately at startup with a `FileNotFoundError` on `train.txt`.

## 4. ACDC / polyp (optional, only if you extend beyond Synapse)

- ACDC preprocessed: <https://drive.google.com/file/d/1CruCQ-jjvA97BX-LIYwXaRMLmp3DN9zc/view> → `./data/ACDC/`
- Polyp splits: <https://drive.google.com/drive/folders/1XyjNgmPqikGxCaOdP0i6Xzf3deDIpbCV> → `./data/polyp/`

`train_synapse_lomix.py` is Synapse-only. Polyp/ACDC training would need the
EMCAD (`train_polyp.py`) or G-CASCADE (`train_ACDC.py`) entry points, which use
the same `lib/` and the same decoder — a reasonable second experiment if the
Synapse runs turn out too slow on this GPU.

## 5. If the GTX 960 turns out too slow

The honest fallback is Colab / Kaggle free tier (T4, 16 GB). Same repo, same
commands, `--amp` still helps. What you must **not** do is report a number you
did not actually produce, or a paper number as if it were your run — the
screening email calls that out explicitly.
