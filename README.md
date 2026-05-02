# DeptBam MobileNet Training

This project trains MobileNet variants (V1/V2/V3) with DeptBAM attention.

## Requirements

- Python 3.8+
- PyTorch + torchvision
- NumPy
- SciPy (required by the Stanford Dogs dataset loader)

Install examples:

```bash
pip install torch torchvision numpy scipy
```

## Dataset

Supported datasets:

- cifar10
- cifar100
- dogs (Stanford Dogs)

For `dogs`, you can download automatically with `--download`.

Set the dataset root with `--data-root`.

## Run Training

Main entrypoint: `DeptBamAttention_main.py`

### MobileNet V1

```bash
python DeptBamAttention_main.py --mobilenet v1 --dataset dogs --data-root D:/Dogs/data
```

### MobileNet V2

```bash
python DeptBamAttention_main.py --mobilenet v2 --dataset dogs --data-root D:/Dogs/data
```

### MobileNet V3 (large)

```bash
python DeptBamAttention_main.py --mobilenet v3 --mobilenet-v3 large --dataset dogs --data-root D:/Dogs/data
```

### MobileNet V3 (small)

```bash
python DeptBamAttention_main.py --mobilenet v3 --mobilenet-v3 small --dataset dogs --data-root D:/Dogs/data
```

### MobileNet V3 with lightweight head

```bash
python DeptBamAttention_main.py --mobilenet v3 --mobilenet-v3 small --mobilenet-v3-lightweight-head --dataset dogs --data-root D:/Dogs/data
```

## Common Options

- `--gpu-id` : GPU index, set `-1` to use CPU.
- `--batch-size` : Batch size.
- `--epochs` : Number of epochs.
- `--learning-rate` : Initial learning rate.
- `--schedule` : LR milestones, e.g. `--schedule 100 150 180`.
- `--download` : Download dataset files.

## Output

Checkpoints and logs are saved under:

```
./checkpoints/<run-name>/
```

`<run-name>` is generated from the selected MobileNet version and options.
