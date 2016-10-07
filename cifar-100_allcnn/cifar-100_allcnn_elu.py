
from pylab import *
import matplotlib.pyplot as plt

caffe_root = '~/bin/caffe-master/'

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P

def allcnn(sname, batch_size):
    '''
    Creates an all-CNN with ELU activations
    '''
    n = caffe.NetSpec()

    n.data, n.label_coarse, n.label_fine = L.HDF5Data(batch_size=batch_size, source=sname, ntop=3)

    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.elu1 = L.ELU(n.conv1, in_place=True)
    n.conv2 = L.Convolution(n.elu1, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.elu2 = L.ELU(n.conv2, in_place=True)
    n.conv3 = L.Convolution(n.elu2, kernel_size=3, num_output=96, stride=2, weight_filler=dict(type='xavier'))
    n.elu3 = L.ELU(n.conv3, in_place=True)
    n.conv4 = L.Convolution(n.elu3, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.elu4 = L.ELU(n.conv4, in_place=True)
    n.conv5 = L.Convolution(n.elu4, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.elu5 = L.ELU(n.conv5, in_place=True)
    n.conv6 = L.Convolution(n.conv5, kernel_size=3, num_output=192, stride=2, weight_filler=dict(type='xavier'))
    n.elu6 = L.ELU(n.conv6, in_place=True)
    n.conv7 = L.Convolution(n.elu6, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.elu7 = L.ELU(n.conv7, in_place=True)
    n.conv8 = L.Convolution(n.elu7, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.elu8 = L.ELU(n.conv8, in_place=True)
    n.conv9 = L.Convolution(n.elu8, kernel_size=1, num_output=10, weight_filler=dict(type='xavier'))
    n.elu9 = L.ELU(n.conv9, in_place=True)

    # n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    # n.conv2 = L.Convolution(n.conv1, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    # n.bn1 = L.BatchNorm(n.conv2)
    # n.elu2 = L.ELU(n.bn1, in_place=True)
    # n.pool1 = L.Pooling(n.elu2, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    # n.drop1 = L.Dropout(n.pool1, dropout_ratio=0.5)
    # n.conv3 = L.Convolution(n.drop1, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    # n.elu3 = L.ELU(n.conv3, in_place=True)
    # n.conv4 = L.Convolution(n.elu3, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    # n.bn2 = L.BatchNorm(n.conv4)
    # n.elu4 = L.ELU(n.bn2, in_place=True)
    # n.pool2 = L.Pooling(n.elu4, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    # n.drop2 = L.Dropout(n.pool2, dropout_ratio=0.5)
    # n.conv5 = L.Convolution(n.drop2, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    # n.elu5 = L.ELU(n.conv5, in_place=True)
    # n.conv6 = L.Convolution(n.elu5, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    # n.elu6 = L.ELU(n.conv6, in_place=True)
    # n.conv7 = L.Convolution(n.elu6, kernel_size=1, num_output=10, weight_filler=dict(type='xavier'))
    # n.elu7 = L.ELU(n.conv7, in_place=True)

    n.pool = L.Pooling(n.elu9, global_pooling=True, pool=P.Pooling.AVE)
    n.flatten = L.Flatten(n.pool)
    n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

with open('all_cnn_train.prototxt','w') as f:
    f.write(str(allcnn('../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 100)))

with open('all_cnn_test.prototxt','w') as f:
    f.write(str(allcnn('../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100)))

caffe.set_device(0)
caffe.set_mode_gpu()

### load the solver and create train and test nets
solver = None
solver = caffe.SGDSolver('cifar-10_solver.prototxt')

# solver.net.forward()
# solver.test_nets[0].forward()
# solver.net.backward()

solver.solve()
