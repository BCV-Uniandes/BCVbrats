# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import scipy.io
import pdb
import math
from settings import gb

# Convolutional layer (Dropout: 2%)
def conv_layer(input_v, size_in, size_out, name="conv"):
	with tf.name_scope(name):
		w = tf.Variable(tf.truncated_normal([3, 3, 3, size_in, size_out], stddev=0.1), name="W")
		b = tf.Variable(tf.zeros([size_out]), name="B")
		conv = tf.nn.conv3d(input_v, w, strides=[1, 1, 1, 1, 1], padding="VALID")
		act = tf.nn.relu(tf.nn.bias_add(conv, b))
		act = tf.nn.dropout(act, gb.conv_drop)
		return act

# Fully connected layer (Dropout: 50%)
def dense_layer(input_v, ch_out, name="conv"):
	with tf.name_scope(name):
		size = input_v.get_shape().as_list()
		channels = size[-1]
		b = tf.Variable(tf.zeros([ch_out]), name="B")
		conv = tf.layers.dense(input_v, units=ch_out, activation=tf.nn.relu)
		act = tf.nn.bias_add(conv, b)
		act = tf.nn.dropout(act, gb.fc_drop)
		return act

# Deconvolutional layer (Dropout: 2%)
def deconv_layer(input_v, factor, name="conv"):
	with tf.name_scope(name):
		filter_s = gb.convolutions[-1]
		shape = tf.shape(input_v)[1]*factor
		out_shape = tf.stack([tf.shape(input_v)[0], shape, shape, shape, filter_s])
		size = factor*2 - factor%2
		strides = [1, factor, factor, factor, 1]
		w = tf.Variable(tf.truncated_normal([size, size, size, filter_s, filter_s], stddev=0.1), name="W")
		b = tf.Variable(tf.zeros([filter_s]), name="B")
		conv = tf.nn.conv3d_transpose(input_v, w, out_shape, strides, padding="SAME")
		act = tf.nn.bias_add(conv, b)
		act = tf.nn.dropout(act, gb.conv_drop)
		return act

# Convolutional layer (Dropout: 2%)
def last_convolution(input_v, size_out, name="conv"):
	with tf.name_scope(name):
		w = tf.Variable(tf.truncated_normal([1, 1, 1, 150, size_out], stddev=0.1), name="W")
		b = tf.Variable(tf.zeros([size_out]), name="B")
		conv = tf.nn.conv3d(input_v, w, strides=[1, 1, 1, 1, 1], padding="SAME")
		act = tf.nn.bias_add(conv, b)
		return act

def create_path(x, num_path):
	"""
	-----------------------------------INPUTS-----------------------------------
	num_path: string that indicates which path we are making.
	gb.convolutions: array with the number of INPUT channels in each convolutional layer
	gb.residuals: array with the layers were the connections will be. The output of 
	these will be combined with the output from 2 layers behind. ex: the output 
	from the 4th layer will be fused with the output from the second.
		res1: layer to be added (in the example was the layer number 2)
	-----------------------------------OUTPUT-----------------------------------
	The result of the last convolution 
	"""
	input_v = x # Patches of the batch
	counter = 0
	for layer in range(1, len(gb.convolutions)): 
		if layer > 1:
			input_v = output
		name = "conv_"  + num_path + '_' + str(layer)
		conv = conv_layer(input_v, gb.convolutions[layer-1], gb.convolutions[layer], name)
		output = conv

		# # Residual connections 
		if layer == gb.residuals[counter]:
			if tf.shape(conv)[-1] != tf.shape(saved)[-1]:
				bc_size = tf.shape(saved)[0]
				im_size = tf.shape(saved)[1]
				fm_size = (tf.shape(conv)[-1] - tf.shape(saved)[-1])/2
				pad_size = [bc_size, im_size, im_size, im_size, fm_size]
				pad = tf.zeros(pad_size, tf.float32)
				saved = tf.concat([pad,saved,pad], -1)
			output = tf.add(conv, saved)
			if counter < (len(gb.residuals) - 1):
				counter += 1
		else:
			output = conv

		res1 = gb.residuals[counter] - 2
		if layer == res1:
			size = tf.shape(conv)[1] - 4
			saved = tf.slice(conv, [0, 2, 2, 2, 0], [-1,size,size,size,-1]) 
	return output

def dense_block(concatenated):
	"""
	residual must be at least 2 to make a residual connection (lower numbers won't 
	have a res1 to combine with)
	"""
	residual = gb.fc_residual
	input_v = concatenated
	res1 = residual - 1
	ch = 150
	for dense in range(1, gb.fc_layers+1):
		if dense > 1:
			input_v = output

		name = "dense_" + str(dense)
		fc_out = dense_layer(input_v, ch, name)

		if residual > 1 and dense == res1:
			saved = fc_out

		if residual > 1 and dense == residual:
			output = tf.add(fc_out, saved)
		else:
			output = fc_out

	return fc_out

def make_placeholders(sizes, name):
	x = [] 
	for i in range(gb.num_paths):
		names = (name + str(i))
		new_shape = [None, None, None, None, gb.num_ch]
		x.append(tf.placeholder(tf.float32, shape=new_shape, name=names))
	return x

def calculate_accuracy(im, lab): 
	im = tf.cast(im, tf.float32)
	equal = tf.cast(tf.equal(im, lab), tf.float32)
	right = tf.reduce_sum(equal)
	total = tf.cast(tf.size(im), tf.float32)
	accuracy = tf.divide(right, total)
	return accuracy

def size_rf_patch(convs, small=gb.min_patches):
	# Sizes and receptive field of the patches
	out = small*gb.downsample[-1] 
	input_v = [out + convs*2] 
	rf = 3 + 2*(convs - 1) # Receptive field
	for i in gb.downsample:
		new = out/i + 2*convs
		input_v.append(new) 
	return [input_v, out], rf

def make_hparam_string(use_two_fc, use_two_conv):
	conv_param = "conv=2" if use_two_conv else "conv=21"
	return "lr_%.0E,%s" % (gb.learning_rate, conv_param) if use_two_conv else "conv=21"

def tensor_names(mode):
	x = []
	for i in range(gb.num_paths):
		x.append('x' + str(i) + ':0')
	if mode == 'train':
		x.append('y:0')
	return x

def test_sizes_patches():
	sizes = [gb.test_patches]
	convs = len(gb.convolutions) - 1
	out = sizes[0] - 2*convs
	for i in gb.downsample:
		new_out = math.ceil(out/float(i))
		new_in = new_out + 2*convs
		sizes.append(int(new_in))
	return [sizes, out]
