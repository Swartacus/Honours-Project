set -e

# draw cifar-10 relu
python ~/bin/caffe-master/python/draw_net.py cifar-10_allcnn/relu/allcnn_relu_train.prototxt cifar-10_allcnn_relu.png
gnuplot cifar-10_allcnn/relu/gnuplot_commands
