from pylab import *
import matplotlib.pyplot as plt

caffe_root = '~/bin/caffe-master/'

import sys, os
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P
import allcnn_createnets as create
from caffe.proto import caffe_pb2


caffe.set_device(0)
caffe.set_mode_gpu()

'''
*******************************************************************************
Create nets
*******************************************************************************
'''
# create relu net:
with open('allcnn_relu_train.prototxt','w') as f:
    f.write(str(create.allcnn_relu('../../data/cifar-10/cifar10_train_lmdb', 100, 'mean.binaryproto')))
with open('allcnn_relu_test.prototxt','w') as f:
    f.write(str(create.allcnn_relu('../../data/cifar-10/cifar10_test_lmdb', 100, 'mean.binaryproto')))

# # create leakyrelu net:
# with open('allcnn_leakyrelu_train.prototxt','w') as f:
#     f.write(str(allcnn_leakyrelu('../../data/cifar-10/cifar10_train_lmdb', 100, 'mean.binaryproto')))
# with open('allcnn_leakyrelu_test.prototxt','w') as f:
#     f.write(str(allcnn_leakyrelu('../../data/cifar-10/cifar10_test_lmdb', 100, 'mean.binaryproto')))
#
# # create sigmoid net:
# with open('allcnn_sig_train.prototxt','w') as f:
#     f.write(str(allcnn_sig('../../data/cifar-10/cifar10_train_lmdb', 100, 'mean.binaryproto')))
# with open('allcnn_sig_test.prototxt','w') as f:
#     f.write(str(allcnn_sig('../../data/cifar-10/cifar10_test_lmdb', 100, 'mean.binaryproto')))
#
# # create tanh net:
# with open('allcnn_tanh_train.prototxt','w') as f:
#     f.write(str(allcnn_tanh('../../data/cifar-10/cifar10_train_lmdb', 100, 'mean.binaryproto')))
# with open('allcnn_tanh_test.prototxt','w') as f:
#     f.write(str(allcnn_tanh('../../data/cifar-10/cifar10_test_lmdb', 100, 'mean.binaryproto')))
#
# # create absval net:
# with open('allcnn_absval_train.prototxt','w') as f:
#     f.write(str(allcnn_absval('../../data/cifar-10/cifar10_train_lmdb', 100, 'mean.binaryproto')))
# with open('allcnn_absval_test.prototxt','w') as f:
#     f.write(str(allcnn_absval('../../data/cifar-10/cifar10_test_lmdb', 100, 'mean.binaryproto')))
#
# # create power net:
# with open('allcnn_power_train.prototxt','w') as f:
#     f.write(str(allcnn_power('../../data/cifar-10/cifar10_train_lmdb', 100, 'mean.binaryproto')))
# with open('allcnn_power_test.prototxt','w') as f:
#     f.write(str(allcnn_power('../../data/cifar-10/cifar10_test_lmdb', 100, 'mean.binaryproto')))
#
# # create bnll net:
# with open('allcnn_bnll_train.prototxt','w') as f:
#     f.write(str(allcnn_bnll('../../data/cifar-10/cifar10_train_lmdb', 100, 'mean.binaryproto')))
# with open('allcnn_bnll_test.prototxt','w') as f:
#     f.write(str(allcnn_bnll('../../data/cifar-10/cifar10_test_lmdb', 100, 'mean.binaryproto')))
#
# # create elu net:
# with open('allcnn_elu_train.prototxt','w') as f:
#     f.write(str(allcnn_elu('../../data/cifar-10/cifar10_train_lmdb', 100, 'mean.binaryproto')))
# with open('allcnn_elu_test.prototxt','w') as f:
#     f.write(str(allcnn_elu('../../data/cifar-10/cifar10_test_lmdb', 100, 'mean.binaryproto')))
#

'''
*******************************************************************************
Create solvers
*******************************************************************************
'''
# set up solvers:
solverSGD = caffe_pb2.SolverParameter()
solverAdaDelta = caffe_pb2.SolverParameter()
solverAdaGrad = caffe_pb2.SolverParameter()
solverAdam = caffe_pb2.SolverParameter()
solverNAG = caffe_pb2.SolverParameter()
solverRMS = caffe_pb2.SolverParameter()

solverSGD.type = "SGD"
solverAdaDelta.type = "AdaDelta"
solverAdaGrad.type = "AdaGrad"
solverAdam.type = "Adam"
solverNAG.type = "Nesterov"
solverRMS.type = "RMSProp"

solvers = [solverSGD, solverAdaDelta, solverAdaGrad, solverAdam, solverNAG, solverRMS]

path = os.path.dirname(os.path.abspath(__file__))

solverAdam.train_net = path + '/allcnn_relu_train.prototxt'
solverAdam.test_net.append(path + '/allcnn_relu_test.prototxt')
solverAdam.test_interval = 500
solverAdam.test_iter.append(100)
solverAdam.max_iter = 10000

solverAdam.base_lr = 0.01
solverAdam.momentum = 0.9
solverAdam.lr_policy = 'step'
solverAdam.gamma = 0.1
solverAdam.stepsize = 1000
solverAdam.display = 1000
solverAdam.snapshot = 5000
solverAdam.snapshot_prefix = 'cifar-10_relu_adam'
solverAdam.solver_mode = caffe_pb2.SolverParameter.GPU

with open(path + '/allcnn_solver_adam.prototxt','w') as f:
    f.write(str(solverAdam))

solver = None
solver = caffe.get_solver(path + '/allcnn_solver_adam.prototxt')
niter = 250
test_interval = niter/10
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))

for it in range(niter):
    solver.step(1)

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

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))
fig.savefig('adam.png')
plt.close(fig)
