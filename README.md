# Probabilistic learning rate scheduler with provable convergence

## Abstract
Learning rate schedulers have shown great success in speeding up the convergence of learning algorithms in practice. However, their convergence to a minimum has not been theoretically proven. This difficulty mainly arises from the fact that, while traditional convergence analysis prescribes to monotonically decreasing (or constant) learning rates, schedulers opt for rates that often increase and decrease through the training epochs. We aim to bridge this gap by proposing a probabilistic learning rate scheduler (PLRS) that does not conform to the monotonically decreasing condition, while achieving provable convergence guarantees. 
To demonstrate the practical effectiveness of our approach, we evaluate it on deep neural networks across both vision and language tasks, showing competitive or superior performance compared to state-of-the-art learning rate schedulers. Specifically, our experiments include (a) image classification on CIFAR-10, CIFAR-100, Tiny ImageNet, and ImageNet-1K using ResNet, WRN, VGG, and DenseNet architectures, and (b) language model fine-tuning on the SQuAD v1.1 dataset with pretrained BERT. Notably, on ImageNet-1K with ResNet-50, our method surpasses the leading knee scheduler by 2.79% in classification accuracy.



### Usage
* The code supports cifar10,cifar100,tinyimagenet,imagenet,SQuAD v1.1.
#### Cifar10  and Cifar100
* To  run cifar 10 and cifar 100 the user is expected to modify the dataset in the trainer.py.
* Replace the lr_scheduler.py in the location
```torch/optim/lr_scheduler.py``` with the lr_scheduler.py in the given repository. You should be able to find your torch directory within your interpreter folder.
* Uncomment the models that you wish to run with modified checkpoints and uncomment the lr_scheduler that you wish to run the code with. The hyperparameters are within the code the user is not expected to change to replicate the same results in the paper.
* Run
```
    chmod +x run.sh
   ./run.sh
```
#### Tiny imagenet
* To  run tiny imagenet the user is expected to modify the dataset in the tiny_imagenet_trainer.py.
* Replace the lr_scheduler.py in the location
```torch/optim/lr_scheduler.py``` with the lr_scheduler.py in the given repository. You should be able to find your torch directory within your interpreter folder.
* Uncomment the models that you wish to run with modified checkpoints and uncomment the lr_scheduler that you wish to run the code with. The hyperparameters are within the code the user is not expected to change to replicate the same results in the paper.
* Make sure that the dataset is present in the desired location.
* Run
```
    python tiny_imagenet_trainer.py
```
#### Imagenet-1K
* To  run tiny imagenet the user is expected to modify the dataset in the imagenet_trainer.py.
* Replace the lr_scheduler.py in the location
```torch/optim/lr_scheduler.py``` with the lr_scheduler.py in the given repository. You should be able to find your torch directory within your interpreter folder.
* Uncomment the models that you wish to run with modified checkpoints and uncomment the lr_scheduler that you wish to run the code with. The hyperparameters are within the code the user is not expected to change to replicate the same results in the paper.
* Make sure that the dataset is present in the desired location.
* Run
```
    python imagenet_trainer.py
```

#### BERT
* Run the bert_baseline.ipynb file to run BERT with SQuAD v1.1.
* To work with the new lr modify the **trainer_qa.py** script in the **transformers/examples/pytorch/question-answering/trainer_qa.py** in the transformer library.

### Transformer architecture with fairseq
* Follow the  [https://github.com/facebookresearch/fairseq/blob/main/examples/translation/README.md](documentation) to setup the code
* Modifications to make is to add the uniform_lr_scheduler_fairseq.py in the fairseq library as uniform_lr_scheduer. After which the path will become **lib/python3.1/site-packages/fairseq/optim/lr_scheduler/uniform_lr_scheduler.py**. Fine tunings of the lr scheduler is done in the same script.
* Command to run for uniform lr scheduler:
  CUDA_VISIBLE_DEVICES=0 fairseq-train     data-bin/iwslt14.tokenized.de-en     --arch transformer_iwslt_de_en --share-decoder-input-output-embed     --optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.0     --lr 3e-4 --lr-scheduler uniform      --dropout 0.3 --weight-decay 0.0001     --criterion label_smoothed_cross_entropy --label-smoothing 0.1     --max-tokens 4096     --eval-bleu     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'     --eval-bleu-detok moses     --eval-bleu-remove-bpe     --eval-bleu-print-samples     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --max-epoch 50.
  
