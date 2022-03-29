set -e

pid=$$

mkdir ${pid}_output && cd ${pid}_output 

CUDA_VISIBLE_DEVICES=0 python3.7 ../test_ALT.py -c ppcls/configs/ImageNet/ResNet/ResNet50.yaml $1

echo ""
echo "[DONE] Everything is in ${pid}_output directory."
echo ""
echo "ls -l ${pid}_output"
echo ""
cd .. && ls -l ${pid}_output
