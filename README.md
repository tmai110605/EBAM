# Enhanced Bottleneck Attention Module Based on Lightweight Spatial Patterns for MobileNets

Abstract - Bottleneck Attention Module (BAM) is one of the most popular attention mechanisms for concentrative learning in CNN-based networks. However, addressing BAM for MobileNets has obtained at modest performance levels since it might be the less discriminative patterns caused by their shallow backbone of perceptive layers. To mitigate this issue, a significant enhancement of BAM is proposed (named EBAM) by simply taking into account lightweight depthwise-based spatial patterns for the branch of spatial-based attention. Concretely, two standard dilated convolutions in the original BAM will be merely replaced by one depthwise function. This simple replacement would conduct two benefits to improve MobileNets as follows. i) Addressing the depthwise-based spatial attentive features ensures a unity of lightweight patterns through MobileNets' backbone. ii) Thanks to the depthwise operation, EBAM would take into account less learnable parameters than its original version. Experimental results for image recognition on various benchmark datasets have verified the prominent efficiency of our proposal. Particularly, the performance of MobileNetV3 has been boosted by up to ~5% on Stanford Dogs.

# EBam MobileNet Training

This project trains MobileNet variants (V1/V2/V3) with EBAM attention.

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

Main entrypoint: `EBamAttention_main.py`

### MobileNet V1

```bash
python EBamAttention_main.py --mobilenet v1 --dataset dogs --data-root D:/Dogs/data
```

### MobileNet V2

```bash
python EBamAttention_main.py --mobilenet v2 --dataset dogs --data-root D:/Dogs/data
```

### MobileNet V3 (large)

```bash
python EBamAttention_main.py --mobilenet v3 --mobilenet-v3 large --dataset dogs --data-root D:/Dogs/data
```
### Run Evaluation From Checkpoint

Use `--evaluate` to run validation only from a saved checkpoint:

```bash
# If checkpoint is in ./checkpoints/dogs_EBAM_v2/checkpoint_epoch0158_62.57
python EBamAttention_main.py --evaluate --mobilenet v2 --checkpoint ./checkpoints/dogs_EBAM_v2/checkpoint_epoch0158_62.57
```
Pretrained Checkpoints:

You can link to a pretrained checkpoint for MobileNet V2 on Stanford Dogs (example):

```text
./checkpoints/dogs_EBAM_v2/checkpoint_epoch0158_62.57
```


Click here for the checkpoint folder and download instructions: [dogs_EBAM_v2 checkpoint](checkpoints/dogs_EBAM_v2/README.md)

Place your checkpoint in the `./checkpoints/<run-name>/` folder and pass its path via `--checkpoint` when running `--evaluate`.

Notes:

- Make sure `--dataset` and model options (`--mobilenet`, `--mobilenet-v3`) match the model used to create the checkpoint.
- Set dataset location with `--data-root` if needed.
- The script prints validation loss, validation accuracy, and total evaluation time.

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

## Project Structure

```
EBAM/
├── EBamAttention_main.py          # Main training script - entry point
├── writeLogAcc.py                     # Logging utility for training progress and results
├── README.md                          # This file
├── model/
│   ├── common.py                      # Common utilities and helper functions
│   ├── datasets.py                    # Dataset loading and preprocessing
│   ├── EBam.py                     # EBAM implementation (Enhanced Bottleneck Attention Module)
│   └── mobilenet_EBam.py           # MobileNet architectures (V1/V2/V3) integrated with EBAM
└── checkpoints/                       # Output directory for trained models and logs
    └── <run-name>/
        ├── checkpoint_epoch*.pth      # Saved model weights
        └── <run-name>.txt             # Training log with timestamps
```

### Key Files

- **EBamAttention_main.py**: Main training script. Handles argument parsing, model building, training loop, and checkpoint saving.
- **model/EBam.py**: Core EBAM (Enhanced Bottleneck Attention Module) implementation with depthwise spatial attention patterns.
- **model/mobilenet_EBam.py**: MobileNet V1, V2, and V3 architectures with integrated EBAM.
- **model/datasets.py**: Data loading and augmentation for CIFAR-10, CIFAR-100, and Stanford Dogs.
- **writeLogAcc.py**: Utility for writing timestamped logs to checkpoint directory.
