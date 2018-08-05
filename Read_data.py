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

def get_list(mode):
	filenames = [[] for _ in range(gb.num_ch)]
	labelnames = []
	for i in range(gb.num_ch):
		name = gb.impath + gb.ends[i]
		with open(name) as f:
			filenames[i] = f.readlines()
		
		filenames[i] = [x.strip() for x in filenames[i]] 
	filenames = map(list, zip(*filenames))

	name = gb.impath + gb.ends[-1]
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
	min_diff = (size_p - size_p%2)/2 + 1

	if difference.min() <= min_diff or any(difference >= (np.asarray(im_size) - min_diff)):   
		# Posible modification: paddding with the size of the difference     
		image  = [np.pad(i, (min_diff), 'constant', constant_values=(0)) for i in image] 
		v1 = voxel
	else:
		v1 = voxel - min_diff

	v1 = v1.astype(int)
	v2 = v1 + size_p
	patch_list = []
	for i in range(channels):
		patch = image[i][v1[0]:v2[0], v1[1]:v2[1], v1[2]:v2[2]]
		if len(np.unique(patch.shape)) > 1:
			pdb.set_trace()
		patch_list.append(patch)
	return np.stack(patch_list, axis=-1)

def read_image(path, label):
	img = nib.load(path)
	im = img.get_data()

	# Binarization of the labels (whole tumor)
	if label == 1 and gb.num_classes == 2:
		image = (image != 0)*1 
	return image	

def make_patches(label, voxel, sizes, image, mode):
	if mode == 'train':
		patches = [[] for _ in range(gb.num_paths+1)]
	else:
		patches = [[] for _ in range(gb.num_paths)]

	for pathway in range(gb.num_paths):
		if pathway == 0:
			input_im = image
			input_voxel = np.asarray(voxel)
		else:
			input_im, input_voxel = downsampled_patches(image, voxel, gb.downsample[pathway-1])
		
		for im in range(len(voxel)):
			temp_im = extract_patch(input_im, input_voxel[im], sizes[0][pathway], gb.num_ch)
			patches[pathway].append(temp_im)

	if mode == 'train':
		temp_l = []
		for i in range(len(voxel)):
			patches[-1].append(np.squeeze(extract_patch([label], np.asarray(voxel[i]), sizes[1], 1)))

	return patches

def voxel_selection(label, image, sizes):
	batch = gb.patches_per_patient/2
	nop = np.argwhere(image[0] == image[0].min())
	sip = np.argwhere(label == 0)
	# Make sure no voxel from the background is chosen
	dims = np.maximum(nop.max(0),sip.max(0))+1
	out = sip[~np.in1d(np.ravel_multi_index(sip.T,dims),np.ravel_multi_index(nop.T,dims))]

	voxel = random.sample(out, batch)
	options = random.sample(np.argwhere(label != 0), gb.patches_per_patient-batch)
	voxel.extend(options)
	random.shuffle(voxel)
	im_patch = make_patches(label, voxel, sizes, image, 'train')
	return im_patch

def input_pipeline(image_list, label_list, sizes, subepoch):
	print "----- Extracting patches. This might take a while -----"
	tic = time.clock()
	data = [[] for _ in range(gb.num_paths+1)]
	for patient in range(len(label_list)):
		label = read_image(label_list[patient], 1)
		image = []
		for channel in range(gb.num_ch):
			image.append(read_image(image_list[patient][channel], 0))

		data_t = voxel_selection(label, image, sizes)
		for i in range(len(data)-1):
			data[i].extend(data_t[i])

		if (patient+1) % 10 == 0 and patient > 0:
			print (patient+1)*gb.patches_per_patient, "patches extracted. Total patches:", gb.patches_per_patient*len(label_list)
	
	# Shuffle
	temporal = zip(*data)
	random.shuffle(temporal)
	data = zip(*temporal)
	data = [list(i) for i in data]

	toc = time.clock()
	print "Time elapsed:", toc-tic
	return data 

def voxels_test(size_p, desfase, original):
	min_diff = (size_p - size_p%2)/2
	a1 = range(min_diff + desfase, original[0] + (min_diff-1) + desfase, size_p)
	b1 = range(min_diff + desfase, original[-1] + (min_diff-1) + desfase, size_p) 
	c1 = [a1,b1,b1]
	combinations = list(itertools.product(*c1))
	return combinations

def input_test(image, sizes, patches, combinations):
	# List containing the batch for the test
	batch_size = patches[-1]*patches[0]
	print "----- Extracting patches. This might take a while -----"
	tic = time.clock()
	data = [[] for _ in range(gb.num_paths)]
	for rows in range(len(combinations)/batch_size):
		voxel = combinations[:batch_size]
		combinations.extend(voxel)
		del combinations[:batch_size]

		data_t = make_patches(0, voxel, sizes, image, 'test')

		for i in range(len(data)):
			data[i].extend(data_t[i])

		print (rows+1)*(len(voxel)), "patches extracted"

	toc = time.clock()
	print "Time elapsed:", toc-tic
	return data

def save_predictions(prediction, path, outpath):
	image = nib.load(path) # To obtain the header
	header = image.header
	image_affine = image.affine

	new_pred = nib.Nifti1Image(prediction, image_affine)
	new_pred.set_data_dtype(np.uint8)
	nib.save(new_pred, outpath)

def save_probabilities(probability, path, outpath, name):
	image = nib.load(path) # To obtain the header
	header = image.header
	image_affine = image.affine
	temp = np.split(probability, gb.num_classes, axis=-1)

	for i in range(gb.num_classes):
		new_pred = nib.Nifti1Image(probability, image_affine)
		new_pred.set_data_dtype(np.float32)
		nib.save(new_pred, outpath+'Maps/'+name[:-6]+'_ProbMapClass'+str(i)+'.nii.gz')
