# MMAction2 Notes (Recommended for Recognition Baselines)

For Task A (`coarse` / `fine` recognition), you can simplify implementation by training TSM/SlowFast/VideoMAE under MMAction2 using one framework.

Typical command:

```bash
python tools/train.py <config.py> --work-dir <output_dir>
python tools/test.py <config.py> <checkpoint.pth> --eval top_k_accuracy mean_class_accuracy
```

Use `scripts/export_task_a_clips.py --trim-clips` to produce clip files and annotation manifests first.
