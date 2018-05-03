# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import scipy.io
import pdb
import random
import argparse
from settings import gb
from Create_model import * 
from Read_data import *
from operator import itemgetter
import sys

os.environ['CUDA_VISIBLE_DEVICES']='2'
def train(mode,load):
	with tf.Graph().as_default(): 
		print 'Image input sizes:', gb.train_patches, '\t Downsamples:', gb.downsample

		# Create architecture
		if load != '1':
			shape = [gb.batch_size] + [None]*3
			x = make_placeholders('x', shape+[gb.num_ch])
			y = tf.placeholder(tf.float32, shape=shape, name='y')
			x.append(y)
			phase = tf.Variable(True, name = 'phase')

			# Create the pathways
			outputs = [[] for _ in range(gb.num_paths)]
			for pathway in range(gb.num_paths):
				with tf.variable_scope('path' + str(pathway)):
					path = create_path(x[pathway], phase)
				outputs[pathway] = path

			# Fully Connected layers
			concatenated = tf.concat(outputs, axis=-1, name='output_paths')
			fc_out = fc_block(concatenated, phase)

			# Classification layer (softmax)
			last_conv = conv_layer(fc_out, gb.num_classes, filters=1, name='final_conv')
			classification = tf.nn.softmax(logits=fc_out, name='softmax')
			prediction = tf.argmax(classification, axis=-1)

			# Summaries (images)
			pred_temp = tf.cast(tf.expand_dims(prediction, axis=-1), tf.float32)
			summary_image(pred_temp, 3, 'Prediction')
			y_summ = tf.expand_dims(y, axis=-1)
			summary_image(y_summ, 3, 'label')
			summary_image(x[0], 3, 'In')

			# Loss
			y_temp = tf.cast(y, tf.int32)
			loss_temp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_temp, logits=fc_out, name='loss')
			loss = tf.reduce_mean(loss_temp)
			tf.summary.scalar('loss', loss)

			# Optimizer
			learning_rate = tf.Variable(gb.learning_rate, name = 'learning_rate')
			optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, name='optimizer')
			# optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.6, epsilon=10e-4).minimize(loss, name='optimizer')
			print '----- Architecture Created -----'
			# tr_par = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
			# print 'Number of trainable parameters: ', tr_par

			accuracy = calculate_accuracy(prediction, y)
			tf.summary.scalar('accuracy', accuracy)

		with tf.Session() as sess:
			init = (tf.global_variables_initializer(), tf.local_variables_initializer())
			last = 0 # Epochs until now

			# Restore model (if asked)
			if load == '1':
				graph, x, last = restore_model(mode, sess)
				loss = graph.get_tensor_by_name('Mean:0')
				optimizer = graph.get_operation_by_name('optimizer')
				accuracy = graph.get_tensor_by_name('truediv:0')
				learning_rate = graph.get_tensor_by_name('learning_rate:0')
				phase = graph.get_tensor_by_name('phase:0')

			saver = tf.train.Saver()
			summ = tf.summary.merge_all()
			writer = tf.summary.FileWriter(gb.LOGDIR + 'summary')
			sess.run(init)
			writer.add_graph(sess.graph)

			image_list, label_list = get_list(mode) # Volumes to read
			lista = range(gb.num_cases)
			num_it = (gb.factor*gb.patches_per_patient)/gb.batch_size # Iterations per sub-epoch

			# In case a model was restored
			count = last*gb.sub_epochs*num_it # Number of iterations until now 
			num_subepoch = last*gb.sub_epochs # Number of sub-epochs until now
			epochs_to_train = gb.num_epochs-last # Number of epochs to go

			for epoch in range(epochs_to_train):
				last += 1
				# Learning rate decay at every given epoch (gb.lr_decay)
				if last in gb.lr_decay:
					# Deberia pasar esto a una función (REVISAR!)
					times = gb.lr_decay.index(last) + 1
					new_lr = gb.learning_rate/(2**times)
					reduce_lr = tf.assign(learning_rate, new_lr)
					sess.run(reduce_lr)
					print 'New learning rate:', sess.run(learning_rate)

				cases = random.sample(lista, gb.factor) # Volumes to be used in this sub-epoch
				for subepoch in range(gb.sub_epochs):
					num_subepoch += 1
					data_subepoch = input_pipeline(itemgetter(*cases)(image_list), itemgetter(*cases)(label_list))

					print '----- Starting Epoch', last, '- Sub-epoch', subepoch+1, '-----'

					acc_total = 0
					cost_total = 0
					if len(data_subepoch[0]) % gb.batch_size != 0:
						extra = gb.batch_size - (len(data_subepoch[0]) % gb.batch_size)
						data_subepoch = [data_subepoch[j] + data_subepoch[j][:extra] for j in range(2)]
						num_it += 1

					data = [list_split(data_subepoch[i], num_it) for i in range(gb.num_paths+1)]

					for iteration in range(num_it):
						summ2, cost, acc, _ = sess.run([summ, loss, accuracy, optimizer], 
							 feed_dict={i: np.squeeze(d) for i, d in zip(x, [data[j][iteration] for j in range(gb.num_paths+1)])})

						acc_total += acc
						cost_total += cost
						count += 1

						if (iteration+1) % 10 == 0:
							writer.add_summary(summ2, count)
							print 'Iteration:', iteration+1, '\tloss:', cost, '\taccuracy:', acc
					print 'Metrics sub-epoch:', subepoch+1, 'loss:', cost_total/num_it, 'accuracy:', acc_total/num_it
				saver.save(sess, os.path.join(gb.LOGDIR, 'model.ckpt'), last)
				print 'Model', last, 'saved.'

			print('Done training -- epoch limit reached')
			saver.save(sess, os.path.join(gb.LOGDIR, 'model.ckpt'), last)
			sess.close()
			print '\n'+'Training complete!'
# --------------------------------------TEST--------------------------------------
def test(mode):
	if not os.path.exists(gb.outpath):
		os.makedirs(gb.outpath)
		os.makedirs(gb.outpath+'Maps/')

	with tf.Graph().as_default():
		print 'Image input sizes:', gb.test_patches
		image_list, _ = get_list(mode)

		with tf.Session() as sess:
			# Load the model to test
			graph, x, _ = restore_model(mode, sess)
			out = graph.get_tensor_by_name('ArgMax:0') 
			probabilities = graph.get_tensor_by_name('softmax:0')
			phase = graph.get_tensor_by_name('phase:0')
			change_phase = tf.assign(phase, False)
			sess.run(change_phase)
			tr_par = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
			print 'Number of trainable parameters: ', tr_par
			# pdb.set_trace()

			for num in range(len(image_list)):
				im = image_list[num]
				index = im[0].find('Brats17_')
				number = im[0][index:-13]

				image = []
				for channel in range(gb.num_ch):
					temp = read_image(im[channel], 0)
					image.append(np.pad(temp, gb.train_patches+2, 'constant', constant_values=(0)))

				original = temp.shape # Shape of the case to evaluate
				patches = [int(math.ceil(p/float(gb.test_patches))) for p in original] # Number of patches to extract per dimension
				combinations = voxels_test(gb.test_patches, gb.train_patches+2, original)
				data_test = input_test(image, patches, combinations)

				# Prediction and probabilities of the image (Revisar extra!!!)
				total_patches = np.prod(np.array(patches))
				module = (total_patches % gb.batch_size)
				if module != 0:
					extra = gb.batch_size - module
					for i in range(gb.num_paths):
						data_test[i] = np.concatenate((data_test[i], [np.zeros_like(data_test[i][0])]*extra))

				batches = len(data_test[0])/gb.batch_size
				data = [np.split(data_test[i], batches) for i in range(gb.num_paths)]
				prediction = []
				probability = []
				for batch in range(batches):
					pred, prob = sess.run([out, probabilities], 
						feed_dict={i: d for i, d in zip(x, [data[j][batch] for j in range(gb.num_paths)])})

					prediction.extend(pred)
					probability.extend(prob)

				if module != 0:
					del prediction[-extra:]
					del probability[-extra:]

				# Reconstruction of the image
				difference = gb.train_patches - gb.test_patches
				center = (difference - difference%2)/2 # REVISAR (solo funciona porque train_patches=2*test_patches)
				# center = (gb.test_patches - gb.test_patches%2)/2
				prediction = [parche[center:-center, center:-center, center:-center] for parche in prediction]
				probability = [parche[center:-center, center:-center, center:-center] for parche in probability]

				FinalPred = pred_reconstruction(patches, total_patches, prediction, original)
				image_affine = affine(im[0])
				save_predictions(FinalPred, image_affine, gb.outpath+number+'.nii.gz')

				# FinalProb = prob_reconstruction(patches, total_patches, probability, original)
				# for num_map in range(len(FinalProb)):
				# 	if num_map != 3 and num_map != 0:
				# 		new_dir = gb.outpath+'Maps/'+number+'/'
				# 		if not os.path.exists(new_dir):
				# 			os.makedirs(new_dir)
				# 		name = new_dir+number+'_ProbMapClass'+str(num_map)+'.nii.gz'
				# 		save_predictions(FinalProb[num_map], image_affine, name)

				print 'Prediction and probability maps of patient', number, 'saved.'
				# pdb.set_trace()
			sess.close()
if __name__ == '__main__':
	parser = argparse.ArgumentParser() 
	parser.add_argument('--mode', help='train or test')
	parser.add_argument('--load', help='1 to restore a trained model')
	args = parser.parse_args()
	if args.mode == 'train':
		train(args.mode, args.load)
	else:
		test(args.mode)
