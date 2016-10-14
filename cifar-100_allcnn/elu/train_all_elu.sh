set -e

./train_elu_adadelta.sh
./train_elu_adagrad.sh
./train_elu_adam.sh
./train_elu_nag.sh
./train_elu_rms.sh
./train_elu_SGD.sh 
