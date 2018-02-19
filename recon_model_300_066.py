'''
Created on Feb 19, 2018
author: Valery Vishnevskiy 
ETH Zurich, IBT CMR
valera.vishnevskiy@gmail.com
'''

from random import shuffle
import sys
import tensorflow as tf
import scipy.io as sio
import numpy as np
import VNR_model_complex


'''
python3 VN_model_recon_complex.py path_to_mat_file path_to_ckpt path_mat_out
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

mat = sio.loadmat(sys.argv[1])
test_imgs = np.array(mat['imgs_test'], dtype = np.complex64).transpose([2, 0, 1])[:n_batch, :, :]
test_kspc = np.array(mat['kspc_test'], dtype = np.complex64).transpose([2, 0, 1])[:n_batch, :, :]
test_mask = np.array(mat['masks_test'], dtype = np.complex64).transpose([2, 0, 1])[:n_batch, :, :]
imsz = test_imgs.shape[1:]

vn_model = VNR_model_complex.VN_FFT_2D(n_layers = n_layers, n_flt = n_flt, n_conv = n_conv, im_sz = imsz, flt_responses = flt_responses, \
                               n_batch = n_batch, n_interp_knots = n_interp_knots, interp_type = interp_type, recon_cost = 'ssd', \
                               norm_flt = True, cntr_flt = True, thr_alphas = True)

saver = tf.train.Saver()
config = tf.ConfigProto()
sess = tf.InteractiveSession(config=config)
saver.restore(sess, sys.argv[2])

imrecon = vn_model.run_reconstruction(sess, test_kspc, test_mask)
sio.savemat(sys.argv[3], {'x':imrecon})