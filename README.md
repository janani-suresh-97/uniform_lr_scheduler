# Probabilistic learning rate scheduler with provable convergence

## Abstract

Learning rate schedulers have shown great success in speeding up the convergence of learn-
ing algorithms in practice. However, their convergence to a minimum has not been proven
theoretically. This difficulty mainly arises from the fact that, while traditional convergence
analysis prescribes to monotonically decreasing learning rates, schedulers opt for rates that
often increase and decrease through the training epochs. In this work, we aim to bridge
the gap by proposing a probabilistic learning rate scheduler (PLRS), that does not con-
form to the monotonically decreasing condition, with provable convergence guarantees. In
addition to providing detailed convergence proofs, we also show experimental results where
the proposed PLRS performs competitively as other state-of-the-art learning rate sched-
ulers. Specifically, we show that our scheduler performs the best or close to best using the
CIFAR-100 (on ResNet-110 and DenseNet-121 architectures) and CIFAR-10 (on VGG-16
and WRN-28-10 architectures).


### Results Summary

### Cifar 100

| Model       | Learning Rate Schedule | Training Accuracy (%) | Test Accuracy (%) | Accuracy Drop (%) |
|-------------|-------------------------|-----------------------|-------------------|-------------------|
| ResNet-110  | Cosine                  | 74.22                 | 72.66             | 1.56              |
| ResNet-110  | Knee                    | 75.78                 | 72.39             | 2.96              |
| ResNet-110  | One-cycle               | 71.09                 | 70.05             | 1.19              |
| ResNet-110  | Constant                | 69.53                 | 66.67             | 2.51              |
| ResNet-110  | Multi-step              | 63.28                 | 61.20             | 2.39              |
| ResNet-110  | PLRS (ours)             | **77.34**             | **74.61**         | 2.95              |
| DenseNet-40-12| Cosine                  | 82.81                 | 80.47             | 2.07              |
| DenseNet-40-12| Knee                    | 82.81                 | 80.73             | 2.39              |
| DenseNet-40-12| One-cycle               | 73.44                 | 72.39             | 0.90              |
| DenseNet-40-12| Constant                | 82.81                 | 80.73             | 2.39              |
| DenseNet-40-12| Multi-step              | **87.50**             | **84.89**         | 2.39              |
| DenseNet-40-12| PLRS (ours)             | 84.37                 | 83.33             | 0.90              |

### Cifar 10

| **Architecture** | **Scheduler** | **Max Test acc.** | **Mean test acc. (S.D)** |
|------------------|---------------|-------------------|-------------------------|
| VGG-16           | Cosine        | 96.87             | 96.09 (0.78)            |
| VGG-16           | Knee          | 96.87             | **96.35** (0.45)        |
| VGG-16           | One-cycle     | 90.62             | 89.06 (1.56)            |
| VGG-16           | Constant      | 96.09             | 96.06 (0.05)            |
| VGG-16           | Multi-step    | 92.97             | 92.45 (0.90)            |
| VGG-16           | PLRS (ours)   | **97.66**         | 96.09 (1.56)            |
| WRN-28-10        | Cosine        | 92.03             | 91.90 (0.13)            |
| WRN-28-10        | Knee          | **92.04**         | 91.64 (0.63)            |
| WRN-28-10        | One-cycle     | 87.76             | 87.37 (0.35)            |
| WRN-28-10        | Constant      | **92.04**         | **92.00** (0.08)        |
| WRN-28-10        | Multi-step    | 88.94             | 88.80 (0.21)            |
| WRN-28-10        | PLRS (ours)   | 92.02             | 91.43 (0.54)            |

### Usage

* The code supports cifar10 and cifar100 dataset. To change it to cifar 100 the user is expected to modify the datset in the trainer.py.

* Replace the lr_scheduler.py in the location
```torch/optim/``` with the lr_scheduler.py in the given repository. You should be able to find your torch directory within your interpreter folder.

* Uncomment the models that you wish to run with modified checkpoints and uncomment the lr_scheduler that you wish to run the code with. The hyperparameters are within the code the user is not expected to change to replicate the same results in the paper.

* Run

```
    chmod +x run.sh
   ./run.sh
```


