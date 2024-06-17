model="wrn"
# Change the model name to vgg,wrn,densenet,resnet
echo "python -u trainer.py  --save-dir=save$_{model} |& tee -a log_$model"
python -u trainer_cifar.py  --save-dir=save_${model} --epochs 500 |& tee -a log_$model 