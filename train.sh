set -e

pid=$$

mkdir ${pid}_output && cd ${pid}_output 

CUDA_VISIBLE_DEVICES=0 python3.7 ../tools/train.py -c ../ppcls/configs/ImageNet/ResNet/ResNet50.yaml $1

