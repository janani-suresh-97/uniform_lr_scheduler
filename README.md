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

## Test Accuracy: 

![plot](uniform_lr_scheduler/images/test_acc_zoom.png)

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


