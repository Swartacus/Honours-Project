set -e

cd cifa-10/relu/
./train_all_relu.sh
cd ../elu
./train_all_elu.sh
cd ../leakyrelu
./train_all_leakyrelu.sh
