from pylab import *
import matplotlib.pyplot as plt

caffe_root = '~/bin/caffe-master/'

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P

def allcnn(sname, batch_size):
    '''
    Creates a CNN that uses ELUs
    '''
    n = caffe.NetSpec()

    n.data, n.label_coarse, n.label_fine = L.HDF5Data(batch_size=batch_size, source=sname, ntop=3)
    # first stack
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=192, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop1 = L.Dropout(n.pool1, dropout_ratio=0.0)
    n.relu1 = L.ReLU(n.drop1)
    # second stack
    n.conv2 = L.Convolution(n.relu1, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.conv3 = L.Convolution(n.conv2, kernel_size=3, num_output=240, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop2 = L.Dropout(n.pool2, dropout_ratio=0.1)
    n.relu2 = L.ReLU(n.drop2)
    # third stack
    n.conv4 = L.Convolution(n.relu2, kernel_size=1, num_output=240, weight_filler=dict(type='xavier'))
    n.conv5 = L.Convolution(n.conv4, kernel_size=2, num_output=260, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv5, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop3 = L.Dropout(n.pool3, dropout_ratio=0.2)
    n.relu3 = L.ReLU(n.drop3)
    # fourth stack
    n.conv6 = L.Convolution(n.relu3, kernel_size=1, num_output=260, weight_filler=dict(type='xavier'))
    n.conv7 = L.Convolution(n.conv6, kernel_size=2, num_output=280, weight_filler=dict(type='xavier'))
    n.pool4 = L.Pooling(n.conv7, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop4 = L.Dropout(n.pool4, dropout_ratio=0.3)
    n.relu4 = L.ReLU(n.drop4)
    # fifth stack
    n.conv8 = L.Convolution(n.relu4, kernel_size=1, num_output=280, weight_filler=dict(type='xavier'))
    #n.conv9 = L.Convolution(n.conv8, kernel_size=2, num_output=300, weight_filler=dict(type='xavier'))
    n.pool5 = L.Pooling(n.conv6, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop5 = L.Dropout(n.pool5, dropout_ratio=0.4)
    n.relu5 = L.ReLU(n.drop5)
    # sixth stack
    n.conv10 = L.Convolution(n.relu5, kernel_size=1, num_output=300, weight_filler=dict(type='xavier'))
    n.pool6 = L.Pooling(n.conv10, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop6 = L.Dropout(n.pool6, dropout_ratio=0.5)
    n.relu6 = L.ReLU(n.drop6)
    # seventh stack
    n.conv11 = L.Convolution(n.relu6, kernel_size=1, num_output=100, weight_filler=dict(type='xavier'))
    n.pool7 = L.Pooling(n.conv11, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.drop7 = L.Dropout(n.pool7, dropout_ratio=0.0)
    n.relu7 = L.ReLU(n.drop7)

    # n.pool = L.Pooling(n.relu7, global_pooling=True, pool=P.Pooling.AVE)
    # n.flatten = L.Flatten(n.pool)
    # n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.loss = L.SoftmaxWithLoss(n.relu7, n.label)

    return n.to_proto()

with open('elu_cnn_train.prototxt','w') as f:
    f.write(str(allcnn('../../data/cifar-100/cifar_100_caffe_hdf5/train.txt', 100')))

with open('elu_cnn_test.prototxt','w') as f:
    f.write(str(allcnn('../../data/cifar-100/cifar_100_caffe_hdf5/test.txt', 100')))

caffe.set_device(0)
caffe.set_mode_gpu()

### load the solver and create train and test nets
solver = None
solver = caffe.SGDSolver('cifar-10_relu_solver.prototxt')

solver.solve()