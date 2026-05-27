# API

## Train

```powershell
python train_april.py --dataset mosi --gpu 0 --mr 0.3
```

Common options:

```powershell
python train_april.py --dataset mosi --seeds 1111,1112,1113 --gpu 0 --mr 0.3 --model_save_dir ./pt
```

- `--dataset`: `mosi` or `mosei`
- `--seeds`: comma-separated seeds, or `default` for `1111..1119`
- `--gpu`: GPU id; use `-1` for CPU
- `--mr`: missing rate
- `--model_save_dir`: checkpoint directory
- `--config_file`: optional custom config path

## Test

```powershell
python test_april.py --dataset mosi --seed 1111 --gpu 0 --mr 0.3 --model_save_dir ./pt
```

Use the same `--dataset`, `--mr`, and `--model_save_dir` as training.
