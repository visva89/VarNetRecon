'''
Created on Feb 19, 2018
author: Valery Vishnevskiy 
ETH Zurich, IBT CMR
valera.vishnevskiy@gmail.com
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
 
 
class VN_FFT_2D():
    __layers = [] 

    __projection_ops = []

    # placeholders
    __ph_kspc_in = []
    __ph_img_out = []
    __ph_mask = []
    
    __params = {}

    __interp_grid = None

    # fixed and global
    __n_batch = [] # because batch size is fixed now
    __im_sz = []
    __alpha_init_std = 0.03
    __deps = 1e-8

    __xs = [] # reference to reconstruction outputs
    __fnet = [] # final cost


    def __init__(self, n_layers, n_flt, n_conv, im_sz, flt_responses, n_batch = 10, \
                n_interp_knots = 30, interp_type = 'linear', recon_cost = 'ssd', \
                norm_flt = True, cntr_flt = False, thr_alphas = True):
        '''
            n_layers: number of gradient descent iterations to unfold into layers 
            
            n_flt: number of filters in each layer
            
            n_conv: width of convolution
            
            im_sz: size of 2d image that we process
            
            flt_responces: 'maximal' estimated filter responce. 
                We assume conv(x, k) is approximately in [-flt_responces, +flt_responces]
            
            n_batch: batch size. 
                Constant for now, because I thought that it will be faster
            
            n_interp_knots: number of points that is used to model each activation function
            
            interp_type: interpolation that is used to model activation function
                'linear' 
                'cubic'
                'RBF' used in original implementation, memory hungry

            recon_cost: final reconstruction cost
                'ssd' |x-x*|^2_2
                'sad' |x-x*|_1

            norm_flt: add proximal operators to normalize norm of each filter to 1

            cnt_flt: add proximal operators to force each filter to be zero-mean

            thr_alphas: add proximal operators to force alphas to be positive
                 alphas are weights of gradients of data term: alpha* A'*(A*x - b)

        '''
        self.__params = {'n_layers' : n_layers, 'n_flt' : n_flt, 'n_conv' : n_conv, 'im_sz' : im_sz, \
                         'flt_responses' : flt_responses, 'n_batch' : n_batch, 'n_interp_knots' : n_interp_knots, \
                         'interp_type' : interp_type, 'norm_flt' : norm_flt, 'thr_alphas' :thr_alphas}
        self.__im_sz = im_sz
        self.__n_batch = n_batch
        self.__ph_kspc_in = tf.placeholder(tf.complex64, shape = [None, im_sz[0], im_sz[1]])
        self.__ph_mask = tf.placeholder(tf.complex64, shape = [None, im_sz[0], im_sz[1]])
        self.__ph_img_out = tf.placeholder(tf.complex64, shape = [None, im_sz[0], im_sz[1]])        
        self.__xs.append(tf.ifft2d(self.__ph_kspc_in)) # xs[0] <- A^t b
        for i in range(n_layers):
            with tf.name_scope('layer' + str(i + 1)):
                # dx = self.__add_variational_layer(n_flt, n_conv, flt_responses, n_interp_knots, interp_type, \
                #                             self.__xs[i])
                dx = self.__add_variational_layer_mc(n_flt, n_conv, flt_responses, n_interp_knots, interp_type, \
                                            self.__xs[i])
            # the gradient step:
            self.__xs.append(self.__xs[i] - dx) # xs[i+1] <- xs[i] - dx
            self.__layers[-1]['xs'] = self.__xs[-1]
        self.__construct_projection_ops(norm_flt, cntr_flt, thr_alphas)
        self.__set_recon_cost(recon_cost)


    def get_all_model_params(self):
        placeholders = {'kspc' : self.__ph_kspc_in, 'mask' : self.__ph_mask, 'img_out' : self.__ph_img_out}
        net_cost = self.__fnet
        layers = self.__layers
        projection_ops = self.__projection_ops
        return {'placeholders' : placeholders, 'net_cost' : net_cost, 'layers' : layers, \
                'projection_ops' : projection_ops}


    def eval_batch_and_export(self, tf_sess, feed):
        ops = []
        names = []
        # get all parameters
        for i, layer in enumerate(self.__layers):
            for k, v in layer.items():
                key = 'L' + str(i) + k
                ops.append(v)
                names.append(key)
        # evaluate
        vals = tf_sess.run(ops, feed_dict = feed)
        dict_mat = {k : np.array(v)  for k, v in zip(names, vals)}
        return dict_mat


    def eval_batch_and_export_by_layer(self, tf_sess, feed):
        '''
            export all tensors from __layers list of dictionaries
            to mat file for debugging
        '''
        ops = []
        idxs = []
        names = []
        # get all parameters
        for i, layer in enumerate(self.__layers):
            for k, v in layer.items():
                idxs.append(i)
                names.append(k)
                ops.append(v)
        # evaluate
        vals = tf_sess.run(ops, feed_dict = feed)
        dict_mat = {}
        for i in range(len(vals)):
            k = names[i]
            if k not in dict_mat:
                dict_mat[k] = []
            vnpar = np.array(vals[i])
            dict_mat[k].append(vnpar)
        return dict_mat



    def run_reconstruction(self, tf_sess, kspc, mask):
        '''
        kspc : np.complex64 the k-space (zero filled) of size n_batch - sz1 - sz2
        mask : np.float32 undersampling mask of size n_batch - sz1 -sz2}

        parameters of the model should be loaded manually
        note that n_batch should be the same or smaller than during training, because 
        dynamic batch size is not used for now
        '''
        if kspc.shape[0] > self.__n_batch:
            print('Error: reconstruction batch size should be smaller than training')
            raise
        if kspc.shape[0] < self.__n_batch:
            print('Expanding batch')
            N = self.__n_batch - kspc.shape[0]
            z = np.zeros([N, kspc.shape[1], kspc.shape[2]])
            kspc = np.concatenate(kspc, z, axis = 0)
            mask = np.concatenate(mask, z, axis = 0)
        netout = self.__xs[-1]
        return netout.eval(feed_dict = {self.__ph_mask : mask, self.__ph_kspc_in : kspc})


################################################
    def __set_recon_cost(self, recon_cost):
        xout = self.__xs[-1]
        img_out = self.__ph_img_out
        if recon_cost == 'ssd':
            self.__fnet = tf.reduce_sum(tf.square(tf.abs(xout - img_out)))
        elif recon_cost == 'sad':
            self.__fnet = tf.reduce_sum(tf.abs(xout - img_out))
        else:
            print('Unknown reconstruction cost')
            print('Received: ' + '"' + str(recon_cost) + '"')
            raise 


    def __construct_projection_ops(self, norm_flt, cntr_flt, thr_alphas):
        ops = []
        deps = self.__deps
        for layer in self.__layers:
            flts = layer['filters']
            alpha = layer['alpha']
            if cntr_flt:
                op = flts.assign(flts - tf.reduce_mean(flts, axis = [0, 1], keep_dims = True))
                ops.append(op)
            if norm_flt:
                op = flts.assign(flts / tf.sqrt(tf.reduce_sum(tf.square(flts), axis = [0, 1], keep_dims = True) + deps))
                ops.append(op)
            if thr_alphas:
                op = alpha.assign(tf.maximum(alpha, 0.))
                ops.append(op)
        self.__projection_ops = ops


    def __apply_activation(self, flt_responses, n_interp_knots, n_flt, npix, n_batch, Dx, interp_type):
        # on first call we construct the pixel index grid
        if self.__interp_grid is None:
            if interp_type == 'linear' or interp_type == 'cubic':
                # tmp = tf.to_float(tf.range(0, n_flt, name = 'range_op'))
                # self.__interp_grid = tf.tile(tf.reshape(tmp, [1, n_flt]), [npix * tf.shape(Dx)[0], 1], name = 'mtileop')
                self.__interp_grid = tf.constant(np.tile( np.array(range(0, n_flt)).reshape([1, n_flt]), [npix * n_batch, 1]), dtype = tf.float32)
            elif interp_type == 'RBF':
                self.__interp_grid = tf.constant(np.linspace(-flt_responses, flt_responses, n_interp_knots).reshape([1,1, n_interp_knots]), dtype = tf.float32) # 1 - 1 - n_interp_knots

        if interp_type == 'linear':
            fnc_dpsi = lambda Dx: interp_op_linear(-flt_responses, flt_responses, n_interp_knots, n_flt, npix * n_batch, Dx, self.__interp_grid)
        elif interp_type == 'cubic':
            fnc_dpsi = lambda Dx: interp_op_cubic(-flt_responses, flt_responses, n_interp_knots, n_flt, npix * n_batch, Dx, self.__interp_grid)
        elif interp_type == 'RBF':
            fnc_dpsi = lambda Dx: interp_op_RBF(-flt_responses, flt_responses, n_interp_knots, n_flt, npix * n_batch, Dx, self.__interp_grid)
        else:
            print('Unknown interpolation type')
            print('Received: ' + '"' + str(interp_type) + '"')
            raise        
        Dx_flat = tf.reshape(Dx, [-1, n_flt]) # n_batch*sz1*sz2 - n_flt
        psiDx, yK = fnc_dpsi(Dx_flat)
        return psiDx, yK


    def __add_variational_layer(self, n_flt, n_conv, flt_responses, n_interp_knots, interp_type, x_in):
        '''
        uses separate activation for real and imaginary kernels
        '''
        imsz = self.__im_sz
        npix = imsz[0] * imsz[1]
        n_batch = self.__n_batch
        mask_in = self.__ph_mask
        ky = self.__ph_kspc_in

        x_in = tf.reshape(x_in, [-1, imsz[0], imsz[1], 1])
        x_in_R = tf.real(x_in)
        x_in_I = tf.imag(x_in)

        alpha = tf.Variable(tf.abs(tf.truncated_normal([1], stddev=self.__alpha_init_std)) , dtype = tf.float32, name = 'alpha')

        Dstd = 1. / (2 * n_conv * n_conv) # doesn't matter if we normalize
        Ds = tf.Variable(tf.truncated_normal([n_conv, n_conv, 2, n_flt], stddev = Dstd), name = 'filters')
        Ds_I = tf.reshape(Ds[:,:, 0, :], [n_conv, n_conv, 1, n_flt])
        Ds_R = tf.reshape(Ds[:,:, 1, :], [n_conv, n_conv, 1, n_flt])
        Dx_I = tf.nn.conv2d(x_in_I, Ds_I, strides = [1,1,1,1], padding = 'SAME', name = 'before_interp_I') 
        Dx_R = tf.nn.conv2d(x_in_R, Ds_R, strides = [1,1,1,1], padding = 'SAME', name = 'before_interp_R') 
        #Dx: Nbatch - sz1 - sz2 - n_flt

        psiDx_R, yK_R = self.__apply_activation(flt_responses, n_interp_knots, n_flt, npix, n_batch, Dx_R, interp_type)
        psiDx_I, yK_I = self.__apply_activation(flt_responses, n_interp_knots, n_flt, npix, n_batch, Dx_I, interp_type)

        psiDx_R = tf.reshape(psiDx_R, [-1, imsz[0], imsz[1], n_flt], name = 'after_interp_R') 
        psiDx_I = tf.reshape(psiDx_I, [-1, imsz[0], imsz[1], n_flt], name = 'after_interp_I') 
        DtpsiDx_R = tf.nn.conv2d_transpose(psiDx_R, Ds_R, output_shape = [n_batch, imsz[0], imsz[1], 1], strides = [1,1,1,1], name = 'DtpsiDx_R')
        DtpsiDx_I = tf.nn.conv2d_transpose(psiDx_I, Ds_I, output_shape = [n_batch, imsz[0], imsz[1], 1], strides = [1,1,1,1], name = 'DtpsiDx_I')
        DtpsiDx = tf.complex(DtpsiDx_R, DtpsiDx_I)
        xsq = tf.reshape(x_in, [n_batch, imsz[0], imsz[1]])
        AtAxb = tf.ifft2d(mask_in * tf.fft2d(xsq) - ky)
        dxout = tf.cast(alpha, 'complex64') * AtAxb + tf.reshape(DtpsiDx, [-1, imsz[0], imsz[1]]) 

        # store variables and tensor references
        layer = {'alpha' : alpha, 'before_interp_I' : Dx_I, 'before_interp_R' : Dx_R, \
                 'filters' : Ds, \
                 'after_interp_I' : psiDx_I, 'after_interp_R' : psiDx_R, \
                 'interp_knots_I' : yK_I, 'interp_knots_R' : yK_R, \
                 'DtpsiDx_I' : DtpsiDx_I, 'DtpsiDx_R' : DtpsiDx_R, \
                 'dxout' : dxout}
                  # 'filters_I' : Ds_I, 'filters_R' : Ds_R, \
        self.__layers.append(layer)
        return dxout


    def __add_variational_layer_mc(self, n_flt, n_conv, flt_responses, n_interp_knots, interp_type, x_in):
        '''
        uses single activation for real and imaginary kernels which outputs are added
        '''
        imsz = self.__im_sz
        npix = imsz[0] * imsz[1]
        n_batch = self.__n_batch
        mask_in = self.__ph_mask
        ky = self.__ph_kspc_in

        x_in_ch = cplx_to_chn(x_in) # n_batch - imsz1 - imsz2 - 2
        alpha = tf.Variable(tf.abs(tf.truncated_normal([1], stddev=self.__alpha_init_std)) , dtype = tf.float32, name = 'alpha')

        Dstd = 1. / (2 * n_conv * n_conv) # doesn't matter if we normalize
        Ds = tf.Variable(tf.truncated_normal([n_conv, n_conv, 2, n_flt], stddev = Dstd), name = 'filters')
        Dx = tf.nn.conv2d(x_in_ch, Ds, strides = [1,1,1,1], padding = 'SAME', name = 'before_interp') 
        #Dx: Nbatch - sz1 - sz2 - n_flt

        psiDx, yK = self.__apply_activation(flt_responses, n_interp_knots, n_flt, npix, n_batch, Dx, interp_type)
        
        psiDx = tf.reshape(psiDx, [-1, imsz[0], imsz[1], n_flt], name = 'after_interp') 
        DtpsiDx = tf.nn.conv2d_transpose(psiDx, Ds, output_shape = [n_batch, imsz[0], imsz[1], 2], strides = [1,1,1,1], name = 'DtpsiDx')
        DtpsiDx = chn_to_cplx(DtpsiDx)
        xsq = tf.reshape(x_in, [-1, imsz[0], imsz[1]])
        AtAxb = tf.ifft2d(mask_in * tf.fft2d(xsq) - ky)
        dxout = tf.cast(alpha, 'complex64') * AtAxb + tf.reshape(DtpsiDx, [-1, imsz[0], imsz[1]]) 

        # store variables and tensor references
        layer = {'alpha' : alpha, 'filters' : Ds, 'interp_knots' : yK, \
                 'before_interp' : Dx, 'after_interp' : psiDx, 'DtpsiDx' : DtpsiDx, 
                 'dxout' : dxout}
        self.__layers.append(layer)
        return dxout


def chn_to_cplx(x):
    # x: n_batch - sz1 - sz2 - 2 of float32
    return tf.complex(x[:, :, :, 0], x[:, :, :, 1])


def cplx_to_chn(x):
    # x: n_batch - sz1 - sz2  of complex64
    return tf.stack([tf.real(x), tf.imag(x)], 3)


def interp_op_linear(minx, maxx, n_interp_knots, n_flt, n_pix, xin, interp_grid = None):
    # extrapolate?
    # xin:  n_pix - n_flt
    w = (maxx - minx) / (n_interp_knots - 1)
    yK = tf.Variable(tf.truncated_normal([n_interp_knots, n_flt], stddev = 0.01), dtype = tf.float32, name = 'interp_knots')
    xS = (xin - minx) / w
    xS = tf.clip_by_value(xS, 0, n_interp_knots - 0.001)
    xF = tf.floor(xS)
    k = xS - xF
    idx_f = xF
    idx_c = xF + 1
    if interp_grid is None:
        interp_grid = tf.constant(np.tile( np.array(range(0, n_flt)).reshape([1, n_flt]), [n_pix, 1]), dtype = tf.float32)
    # tf.stack moves int32 to CPU, so use int(stack()) not stack(int())
    nd_idx1 = tf.to_int32( tf.stack([idx_f, interp_grid], 2) )
    nd_idx2 = tf.to_int32( tf.stack([idx_c, interp_grid], 2) )
    y_f = tf.gather_nd(yK, nd_idx1)
    y_c = tf.gather_nd(yK, nd_idx2)
    y = y_f * (1 - k) + k * y_c
    return y, yK


def interp_op_cubic(minx, maxx, n_interp_knots, n_flt, n_pix, xin, interp_grid = None):
    # extrapolate?
    # xin:  n_pix - n_flt
    w = (maxx - minx) / (n_interp_knots - 1)
    yK = tf.Variable(tf.truncated_normal([n_interp_knots, n_flt], stddev = 0.01), dtype = tf.float32, name = 'interp_knots')
    xS = (xin - minx) / w
    xS = tf.clip_by_value(xS, 0, n_interp_knots - 0.001)
    xF = tf.floor(xS)
    k = xS - xF
    idx_1 = tf.clip_by_value(xF - 1, 0, n_interp_knots - 1)
    idx_2 = tf.clip_by_value(xF - 0, 0, n_interp_knots - 1)
    idx_3 = tf.clip_by_value(xF + 1, 0, n_interp_knots - 1)
    idx_4 = tf.clip_by_value(xF + 2, 0, n_interp_knots - 1)
    if interp_grid is None:
        interp_grid = tf.constant(np.tile( np.array(range(0, n_flt)).reshape([1, n_flt]), [n_pix, 1]), dtype = tf.float32)
    nd_idx1 = tf.to_int32( tf.stack([idx_1, interp_grid], 2) )
    nd_idx2 = tf.to_int32( tf.stack([idx_2, interp_grid], 2) )
    nd_idx3 = tf.to_int32( tf.stack([idx_3, interp_grid], 2) )
    nd_idx4 = tf.to_int32( tf.stack([idx_4, interp_grid], 2) )

    y_1 = tf.gather_nd(yK, nd_idx1)
    y_2 = tf.gather_nd(yK, nd_idx2)
    y_3 = tf.gather_nd(yK, nd_idx3)
    y_4 = tf.gather_nd(yK, nd_idx4)

    k1 = k * ((2 - k) * k - 1) 
    k2 = (k * k * (3 * k - 5) + 2) 
    k3 = k * ((4 - 3 * k) * k + 1)
    k4 = k * k * (k - 1) 
    y = (k1 * y_1 + k2 * y_2 + k3 * y_3 + k4 * y_4) * 0.5
    return y, yK    


def interp_op_RBF(minx, maxx, n_interp_knots, n_flt, n_pix, xin, interp_grid = None):
    # xin:  n_pix - n_flt
    sigma = (maxx - minx) / (n_interp_knots - 1)
    yK = tf.Variable(tf.truncated_normal([n_flt, n_interp_knots], stddev = 0.01), dtype = tf.float32, name = 'interp_knots')
    if interp_grid is None:
        interp_grid = tf.constant(np.linspace(minx, maxx, n_interp_knots).reshape([1,1, n_interp_knots]), dtype = tf.float32) # 1 - 1 - n_interp_knots
    xin = tf.reshape(xin, [n_pix, n_flt, 1]) # reshape for broadcasting
    pw_dst = tf.square((xin - interp_grid))
    # pw_dst = xin**2 + interp_grid**2 - (2 * xin * interp_grid) # npix - n_flt - n_interp_knots
    rbf_weights = tf.exp( -pw_dst / (2 * sigma * sigma) )
    y = tf.reduce_sum(rbf_weights * tf.reshape(yK, [1, n_flt, n_interp_knots]), axis = 2, keep_dims = False)
    return y, yK    




