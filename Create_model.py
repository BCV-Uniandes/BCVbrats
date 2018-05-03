# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import scipy.io
import pdb
import math
from settings import gb

# ------ Layers ------
def weight_variable_msra(shape, name):
	return tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.variance_scaling_initializer())

def conv_layer(input_v, size_out, filters, strides=[1]*5, padding='SAME',name='conv3d'):
	with tf.variable_scope(name):
		in_features = input_v.get_shape().as_list()[-1]
		kernel = weight_variable_msra([filters]*3 + [in_features, size_out], name=name+'kernel')
		output = tf.nn.conv3d(input_v, kernel, strides, padding, name=name)
	return output

def fc_layer(input_v, ch_out, phase, name='fc'):
	with tf.variable_scope(name):
		conv = tf.layers.dense(input_v, use_bias=True, bias_initializer=tf.zeros_initializer(), units=ch_out, activation=tf.nn.relu)
		act = dropout(conv, gb.fc_drop, phase, name=name+'out')
		tf.summary.histogram('activations', act)
	return act

def deconv_layer(input_v, factor, phase, name='deconv'):
	with tf.variable_scope(name):
		in_sizes = input_v.get_shape().as_list()
		shape = tf.shape(input_v)[1]*factor
		out_shape = tf.stack([in_sizes[0], shape, shape, shape, in_sizes[-1]])
		size = factor*2 - factor%2
		strides = [1, factor, factor, factor, 1]
		w = weight_variable_msra([size]*3 + [in_sizes[-1]]*2, name=name+'_kernel')
		conv = tf.nn.conv3d_transpose(input_v, w, out_shape, strides, padding='SAME', name=name)
		act = dropout(conv, gb.conv_drop, phase, name=name+'out')
	return act

def transition_layer(_input, phase, pool_depth=2, name='trans'):
	with tf.variable_scope(name):
		out_features = int(int(_input.get_shape()[-1]) * gb.reduction)
		output = tf.nn.relu(_input)
		output = conv_layer(output, out_features, filters=1, name=name+'_comp')
		output = pool(output, k=2, d=pool_depth, name=name)
	return output

def pool(_input, k, d=2, width_k=None, type='max', k_stride=None, d_stride=None, k_stride_width=None,name='pool'):
	ksize = [1, d, d, d, 1]
	strides = [1, k, k, k, 1]
	padding = 'SAME'
	if type is 'max':
		output = tf.nn.max_pool3d(_input, ksize, strides, padding, name=name+'_pool')
	elif type is 'avg':
		output = tf.nn.avg_pool3d(_input, ksize, strides, padding, name=name+'_pool')
	else:
		output = None
	return output

def dropout(_input, prob, phase, name):
	if prob < 1:
		output = tf.cond(phase, lambda: tf.nn.dropout(_input, prob, name=name), lambda: _input)
	else:
		output = _input
	return output

# Architecture

def create_path(x, phase):
	net = conv_layer(x, gb.convolutions[0][0], filters=1, name='conv0')
	if gb.cumulative:
		conv = Cumulative(net, phase)
	else:
		conv = Progressive(net, name, phase)
	return conv

def Cumulative(input_v, phase):
	output = input_v
	upsample = [None]*gb.total_blocks
	for block in range(gb.total_blocks):
		with tf.variable_scope('denseBlock' + str(block)):
			output = dense_block(output, phase)
		if block != 0:
			factor = 2**block
			temp = deconv_layer(output, factor, phase, 'deconv'+str(block))
			# upsample[block] = verify_shape(temp, input_v)
		else:
			upsample[block] = output # tenía input_v. Corrección: 30/04/2018 - 21:35 (último: train10)

		output = transition_layer(output, phase, 2, name='trn'+str(block))
	output = tf.concat(upsample, axis=-1)
	return output

def Progressive(input_v, phase):
	output = input_v
	for block in range(gb.total_blocks):
		with tf.variable_scope('denseBlock' + str(block)):
			output = dense_block(output, phase)
		if block != (gb.total_blocks - 1):
			output = transition_layer(output, phase, 2, name='trn'+str(block))
	factor = 2**(gb.total_blocks - 1)
	output = deconv_layer(output, factor, phase, 'deconv_final')
	output = verify_shape(output, input_v)
	return output

def dense_block(_input, phase):
	# initial_conv = conv_layer(_input, gb.convolutions[0][0], filters=1, name='initial_conv')
	# output = tf.nn.relu(initial_conv)
	output = _input
	num_layers = len(gb.convolutions)
	for layer in range(num_layers-1):
		output = add_internal_layer(output, layer, phase, name=str(layer))
		# output = conv_layer(output, gb.convolutions[layer+1][0], filters=1, name='_trans'+str(layer))
		# output = tf.nn.relu(output)
		# pdb.set_trace()
	
	output = add_internal_layer(output, -1, phase, name='_int'+str(num_layers))
	return output

def add_internal_layer(_input, layer, phase, name='int'):
	with tf.variable_scope('rnext_' + name):
		comp_out = resnext(_input, gb.convolutions[layer], phase)
		# comp_out = tf.add(comp_out,_input)
	output = tf.concat(axis=4, values=(_input, comp_out))
	return output

def resnext(input_v, sizes, phase):   
    block = [None]*gb.cardinality
    for index in range(gb.cardinality):
        block[index] = tf.nn.relu(input_v)
        block[index] = conv_layer(block[index], sizes[0], filters=1, name='card'+str(index)+'_0')

        block[index] = tf.nn.relu(block[index])
        block[index] = conv_layer(block[index], sizes[1], filters=3, name='card'+str(index)+'_1')
    merge = tf.concat(block, axis=-1)
    nextUnit = tf.nn.relu(merge)
    nextUnit = conv_layer(nextUnit, sizes[2], filters=1, name='card_2')
    return nextUnit 

def fc_block(concatenated, phase):
	ch = 150
	with tf.variable_scope('fc_block'):
		for fc in range(1, gb.fc_layers+1):
			name = 'fc' + str(fc)
			conv = fc_layer(concatenated, ch, phase, name)
			concatenated = conv
		# conv = fc_layer(concatenated, gb.num_classes, phase, 'last_fc')
	return concatenated

# ------ Others ------
def verify_shape(upsample, original):
	# In case the shape had changed due to the input size and number of blocks ¡¡REVISAR!!
	outSizes = upsample.get_shape().as_list()
	dif = tf.abs(tf.shape(original)[1] - tf.shape(upsample)[1])
	if dif != 0:
		out_shape = tf.stack([outSizes[0], tf.shape(original)[1], tf.shape(original)[1], tf.shape(original)[1], outSizes[-1]])
		output = tf.slice(upsample, [0, dif, dif, dif, 0], out_shape)
		output.set_shape([outSizes[0],None,None,None,outSizes[-1]])
	else:
		output = upsample
	return output

def calculate_accuracy(im, lab): 
	im = tf.cast(im, tf.float32)
	equal = tf.cast(tf.equal(im, lab), tf.float32)
	right = tf.reduce_sum(equal)
	total = tf.cast(tf.size(im), tf.float32)
	accuracy = tf.divide(right, total)
	return accuracy

def summary_image(im, corte, name):
	ind = im[3:10:3,corte,:,:,0]
	ind = tf.expand_dims(ind, -1)
	tf.summary.image(name + str(0), ind)

def make_placeholders(name, shape):
	x = []
	for i in range(gb.num_paths):
		names = (name + str(i))
		x.append(tf.placeholder(tf.float32, shape=shape, name=names))
	return x

def restore_model(mode, sess):
	ckpt = tf.train.get_checkpoint_state(gb.LOGDIR)
	loadpath = ckpt.model_checkpoint_path
	saver = tf.train.import_meta_graph(loadpath + '.meta')
	graph = tf.get_default_graph()  
	saver.restore(sess, loadpath)
	x = tensor_names(mode)

	index = loadpath.find('ckpt-')+5
	last = int(loadpath[index:])
	print '------------ Model', last, 'restored ------------', gb.LOGDIR
	return graph, x, last

def tensor_names(mode):
	x = []
	for i in range(gb.num_paths):
		x.append('x' + str(i) + ':0')
	if mode == 'train':
		x.append('y:0')
	return x
