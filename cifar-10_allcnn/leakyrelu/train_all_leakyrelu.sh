set -e

./train_leakyrelu_adadelta.sh
./train_leakyrelu_adagrad.sh
./train_leakyrelu_adam.sh
./train_leakyrelu_nag.sh
./train_leakyrelu_rms.sh
./train_leakyrelu_SGD.sh 
