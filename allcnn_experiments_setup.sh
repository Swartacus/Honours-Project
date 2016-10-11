set -e

cd cifar-10_allcnn
python allcnn_make.py
cd ../cifar-100_allcnn
python allcnn_make.py
cd ../mnist_allcnn
python allcnn_make.py
