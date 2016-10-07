
from pylab import *
import matplotlib.pyplot as plt

caffe_root = '~/bin/caffe-master/'

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P

def allcnn(lmdb, batch_size, mean):
    '''
    Creates an all-CNN
    '''
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,transform_param=dict(mean_file=mean), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.conv2 = L.Convolution(n.relu1, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.conv2, in_place=True)
    n.conv3 = L.Convolution(n.relu2, kernel_size=3, num_output=96, stride=2, weight_filler=dict(type='xavier'))
    n.relu3 = L.ReLU(n.conv3, in_place=True)
    n.conv4 = L.Convolution(n.relu3, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.relu4 = L.ReLU(n.conv4, in_place=True)
    n.conv5 = L.Convolution(n.relu4, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.relu5 = L.ReLU(n.conv5, in_place=True)
    n.conv6 = L.Convolution(n.conv5, kernel_size=3, num_output=192, stride=2, weight_filler=dict(type='xavier'))
    n.relu6 = L.ReLU(n.conv6, in_place=True)
    n.conv7 = L.Convolution(n.relu6, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    n.relu7 = L.ReLU(n.conv7, in_place=True)
    n.conv8 = L.Convolution(n.relu7, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    n.relu8 = L.ReLU(n.conv8, in_place=True)
    n.conv9 = L.Convolution(n.relu8, kernel_size=1, num_output=10, weight_filler=dict(type='xavier'))
    n.relu9 = L.ReLU(n.conv9, in_place=True)

    # n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    # n.conv2 = L.Convolution(n.conv1, kernel_size=3, num_output=96, weight_filler=dict(type='xavier'))
    # n.bn1 = L.BatchNorm(n.conv2)
    # n.relu2 = L.ReLU(n.bn1, in_place=True)
    # n.pool1 = L.Pooling(n.relu2, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    # n.drop1 = L.Dropout(n.pool1, dropout_ratio=0.5)
    # n.conv3 = L.Convolution(n.drop1, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    # n.relu3 = L.ReLU(n.conv3, in_place=True)
    # n.conv4 = L.Convolution(n.relu3, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    # n.bn2 = L.BatchNorm(n.conv4)
    # n.relu4 = L.ReLU(n.bn2, in_place=True)
    # n.pool2 = L.Pooling(n.relu4, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    # n.drop2 = L.Dropout(n.pool2, dropout_ratio=0.5)
    # n.conv5 = L.Convolution(n.drop2, kernel_size=3, num_output=192, weight_filler=dict(type='xavier'))
    # n.relu5 = L.ReLU(n.conv5, in_place=True)
    # n.conv6 = L.Convolution(n.relu5, kernel_size=1, num_output=192, weight_filler=dict(type='xavier'))
    # n.relu6 = L.ReLU(n.conv6, in_place=True)
    # n.conv7 = L.Convolution(n.relu6, kernel_size=1, num_output=10, weight_filler=dict(type='xavier'))
    # n.relu7 = L.ReLU(n.conv7, in_place=True)

    n.pool = L.Pooling(n.relu9, global_pooling=True, pool=P.Pooling.AVE)
    n.flatten = L.Flatten(n.pool)
    n.score = L.InnerProduct(n.flatten, num_output=10, weight_filler=dict(type='gaussian'))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

with open('all_cnn_train.prototxt','w') as f:
    f.write(str(allcnn('../../data/cifar-10/cifar10_train_lmdb', 100, 'mean.binaryproto')))

with open('all_cnn_test.prototxt','w') as f:
    f.write(str(allcnn('../../data/cifar-10/cifar10_test_lmdb', 100, 'mean.binaryproto')))

caffe.set_device(0)
caffe.set_mode_gpu()

### load the solver and create train and test nets
solver = None
solver = caffe.SGDSolver('cifar-10_solver.prototxt')

# solver.net.forward()
# solver.test_nets[0].forward()
# solver.net.backward()

# solver.solve()

niter = 250
test_interval = niter / 10
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data

    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4

_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))
plt.show()
