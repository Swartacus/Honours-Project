set -e

cd cifar-10_allcnn/relu/
./train_all_relu.sh
cd ../elu
./train_all_elu.sh
cd ../leakyrelu
./train_all_leakyrelu.sh
