set -e

#cd cifar-100_allcnn/relu/
#./train_all_relu.sh
cd cifar-100_allcnn/elu
./train_all_elu.sh
cd ../leakyrelu
./train_all_leakyrelu.sh
