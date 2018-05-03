# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import scipy.io
import pdb
import random
import itertools
from settings import gb
from Create_model import *
from scipy.ndimage.interpolation import zoom
import nibabel as nib
import time
import ConfigParser
from joblib import Parallel, delayed

def get_list(mode):
	filenames = [[] for _ in range(gb.num_ch)]
	labelnames = []
	if mode == 'train':
		root = gb.impath
	else:
		root = gb.testpath

	for i in range(gb.num_ch):
		name = root + gb.ends[i]
		with open(name) as f:
			filenames[i] = f.readlines()
		
		filenames[i] = [x.strip() for x in filenames[i]] 
	filenames = map(list, zip(*filenames))

	if mode == 'train':
		name = root + gb.ends[-1]
		with open(name) as f:
			labelnames = f.readlines()
		labelnames = [x.strip() for x in labelnames]
	return filenames, labelnames

def downsampled_patches(image, voxel, downsample):
	factor = 1.0/downsample
	n_voxel = []
	for i in range(len(voxel)):
		n_voxel.append(np.floor((factor*np.asarray(voxel[i]))))
	n_image = []
	for i in range(len(image)):
		image_list = zoom(image[i], order = 0, zoom=factor)
		n_image.append(image_list)
	return n_image, n_voxel

def extract_patch(image, voxel, size_p, channels):
	im_size = image[0].shape
	difference = im_size - voxel - 1
	difference = np.asarray(difference)
	center = (size_p - size_p%2)/2 + 1

	if difference.min() <= center or any(difference >= (np.asarray(im_size) - center)):
		image  = [np.pad(i, center, 'constant', constant_values=(0)) for i in image] 
		v1 = voxel
	else:
		v1 = voxel - center

	v1 = v1.astype(int)
	v2 = v1 + size_p
	patch_list = []
	for i in range(channels):
		patch = image[i][v1[0]:v2[0], v1[1]:v2[1], v1[2]:v2[2]]
		patch_list.append(patch)
	return np.stack(patch_list, axis=-1)

def read_image(path, label):
	img = nib.load(path)
	image = np.squeeze(img.get_data())
	# image = np.transpose(image, [2,1,0])
	return image	

def make_patches(label, voxel, image, size_patches, mode):
	if mode == 'train':
		patches = [[] for _ in range(gb.num_paths+1)]
	else:
		patches = [[] for _ in range(gb.num_paths)]

	input_im = image
	input_voxel = np.asarray(voxel)	
	for pathway in range(gb.num_paths):
		if pathway != 0:
			input_im, input_voxel = downsampled_patches(image, voxel, gb.downsample[pathway-1])
		
		for im in range(len(voxel)):
			temp_im = extract_patch(input_im, input_voxel[im], size_patches, gb.num_ch)
			patches[pathway].append(temp_im)

	if mode == 'train':
		for i in range(len(voxel)):
			patches[-1].append(np.squeeze(extract_patch([label], np.asarray(voxel[i]), size_patches, 1)))
	return patches

def voxel_selection(label, image):
	annot = np.unique(label)
	batch = gb.patches_per_patient/(len(annot))
	extra = gb.patches_per_patient % len(annot) 
	nop = np.argwhere(image[0] == image[0].min()) # Background voxels
	sip = np.argwhere(label == 0)
	# Make sure no voxel from the background is chosen
	dims = np.asarray(image[0].shape)
	out = sip[~np.in1d(np.ravel_multi_index(sip.T,dims),np.ravel_multi_index(nop.T,dims))]
	voxel = random.sample(out, batch)

	for i in annot[1:]:
		recortes = batch
		if extra != 0 and i == annot[1]:
			recortes = batch + extra
		options = random.sample(np.argwhere(label == i), recortes)
		voxel.extend(options)

	random.shuffle(voxel)
	im_patch = make_patches(label, voxel, image, gb.train_patches, 'train')
	return im_patch

def parallel_patches(image_list, label_list):
	label = read_image(label_list, 1)
	image = []
	for channel in range(gb.num_ch):
		image.append(read_image(image_list[channel], 0))
	data_t = voxel_selection(label, image)
	return data_t

def input_pipeline(image_list, label_list):
	# List containing the batches for the train
	print "----- Extracting patches. This might take a while -----"
	tic = time.clock()
	data = [[] for _ in range(gb.num_paths+1)]
	data_temp = Parallel(n_jobs = 10)(delayed(parallel_patches)(image_list[j],label_list[j]) for j in range(len(label_list)))
	for i in range(len(data)):
		for j in range(len(data_temp)):
			data[i].extend(data_temp[j][i])

	# Shuffle
	temporal = zip(*data)
	random.shuffle(temporal)
	data = zip(*temporal)
	data = [list(i) for i in data]

	toc = time.clock()
	print "Time elapsed:", toc-tic
	return data

def voxels_test(size_p, desfase, original):
	center = (size_p - size_p%2)/2
	a1 = range(center + desfase + 1, original[0] + center + desfase, size_p)
	b1 = range(center + desfase + 1, original[-1] + center + desfase, size_p) 
	c1 = [a1,a1,b1] # [a1,b1,b1]
	combinations = list(itertools.product(*c1))
	return combinations

def parallel_test(voxel, image):
	data_t = make_patches(0, voxel, image, gb.train_patches, 'test')
	return data_t

def input_test(image, patches, combinations):
	# List containing the batches for the test
	tic = time.clock()
	print "----- Extracting patches. This might take a while -----"
	data = [[] for _ in range(gb.num_paths)]
	voxel = list_split(combinations,patches[0])
	data_temp = Parallel(n_jobs = 10)(delayed(parallel_test)(j, image) for j in voxel)
	for i in range(len(data)):
		for j in range(len(data_temp)):
			data[i].extend(data_temp[j][i])

	toc = time.clock()
	print "Time elapsed:", toc-tic
	return data

def pred_reconstruction(patches, total, image, original):
	temporal = list_split(image, patches[0])
	axis1 = []
	for i in range(patches[0]):
		axis2 = list_split(temporal[i], patches[1])
		temp_ax1 = [np.concatenate(axis2[i], axis=2) for i in range(patches[1])]
		axis1.append(np.concatenate(temp_ax1, axis=1))
	axis0 = np.concatenate(axis1, axis=0)
	output = axis0[:original[0],:original[1],:original[2]]
	return output

def prob_reconstruction(patches, total, image, original):
	new_im = np.squeeze(image)
	output = []
	for j in range(5):
		output.append(pred_reconstruction(patches, total, new_im[:,:,:,:,j], original))
	return output

def list_split(lista, parts):
	size_parts = len(lista)/parts
	output = [lista[i:i+size_parts] for i in range(0, len(lista), size_parts)]
	return output

def affine(path):
	image = nib.load(path)
	image_affine = image.affine
	return image_affine

def save_predictions(prediction, image_affine, outpath):
	# prediction = np.transpose(prediction, [2,1,0])
	new_pred = nib.Nifti1Image(prediction, image_affine)
	new_pred.set_data_dtype(np.uint8)
	nib.save(new_pred, outpath)