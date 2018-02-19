'''
Created on Feb 19, 2018
author: Valery Vishnevskiy 
ETH Zurich, IBT CMR
valera.vishnevskiy@gmail.com
'''

import os
from random import shuffle
import random
import string
import sys
import tensorflow as tf
import scipy.io as sio
import numpy as np
import time 
import VNR_model_complex


'''
python3 learn_model_300_066.py path_mat_input path_to_ckpt path_mat_output

'''

n_batch = 11 
n_flt = 30
n_conv = 7
n_layers = 10
flt_responses = 3.5
n_interp_knots = 35
interp_type = 'cubic'
learning_rate0 = 1e-3
imsim = 'ssd'
max_iters = 10000


path_mat_in = sys.argv[1]
path_ckpt = sys.argv[2]
path_mat_out = sys.argv[3]

mat = sio.loadmat(path_mat_in)
imgs_t = np.array(mat['imgs_train'], dtype = np.complex64)
kspc = np.array(mat['kspc_train'], dtype = np.complex64)
masks = np.array(mat['masks_train'], dtype = np.complex64)
imgs_t = np.transpose(imgs_t, [2, 0, 1])
kspc = np.transpose(kspc, [2, 0, 1])
masks = np.transpose(masks, [2, 0, 1])
test_imgs = np.array(mat['imgs_test'], dtype = np.complex64).transpose([2, 0, 1])[:n_batch, :, :]
test_kspc = np.array(mat['kspc_test'], dtype = np.complex64).transpose([2, 0, 1])[:n_batch, :, :]
test_mask = np.array(mat['masks_test'], dtype = np.complex64).transpose([2, 0, 1])[:n_batch, :, :]
imsz = imgs_t.shape[1:]


print('Shapes, ', imgs_t.shape, ' ', kspc.shape, masks.shape, ' test', test_kspc.shape)
vn_model = VNR_model_complex.VN_FFT_2D(n_layers = n_layers, n_flt = n_flt, n_conv = n_conv, im_sz = imsz, flt_responses = flt_responses, \
                               n_batch = n_batch, n_interp_knots = n_interp_knots, interp_type = interp_type, recon_cost = imsim, \
                               norm_flt = True, cntr_flt = True, thr_alphas = True)
params = vn_model.get_all_model_params()
projection_ops = params['projection_ops']
ph_kspc = params['placeholders']['kspc']
ph_mask = params['placeholders']['mask']
ph_img_out = params['placeholders']['img_out']
net_cost = params['net_cost']
ph_learning_rate = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(ph_learning_rate).minimize(net_cost)


lr = learning_rate0
print('LR0: ', lr)
fvals_test = []
fvals_train = []
feed_test = {ph_kspc : test_kspc, ph_mask : test_mask, ph_img_out: test_imgs}

saver = tf.train.Saver()
config = tf.ConfigProto()
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())
sess.run(projection_ops) 
for i in range(max_iters+1):
    # select batch            
    idx0 = np.random.randint(imgs_t.shape[0] - n_batch)
    bimgs = imgs_t[idx0:idx0 + n_batch, :, :]
    bkspc = kspc[idx0:idx0 + n_batch, :, :]
    bmask = masks[idx0:idx0 + n_batch, :, :]

    feed_train = {ph_kspc : bkspc, ph_mask : bmask, ph_img_out : bimgs, ph_learning_rate : lr}
    if i == 5000:
        lr = lr * 0.8
    if i % 100 == 0: # print test 
        test_accuracy = net_cost.eval(feed_dict = feed_test)
        fvals_test.append(test_accuracy)
        print("step %d, test recon err %.5f"%(i, test_accuracy))
        if i:
            print('Time ', time.time() - tm)
        tm = time.time()
    if i%1000 == 0: # store results
        # dict_mat = vn_model.eval_batch_and_export(sess, feed_test)
        dict_mat = vn_model.eval_batch_and_export_by_layer(sess, feed_test)
        dict_mat['fv_train'] = np.array(fvals_train, dtype = np.float32)
        dict_mat['fv_test'] = np.array(fvals_test, dtype = np.float32)
        dict_mat['learning_rate'] = np.array(lr, dtype = np.float32)
        sio.savemat(path_mat_out, dict_mat)
        save_path = saver.save(sess, path_ckpt)
        print("Model saved in path: %s" % save_path)
 
    # optimization steps
    ftr, _ = sess.run([net_cost, train_step], feed_dict = feed_train)
    sess.run(projection_ops) 
    fvals_train.append(ftr)

