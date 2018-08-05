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

def train(mode,load):
	with tf.Graph().as_default(): 
		hparam = make_hparam_string(False, False)
		sizes, r_field = size_rf_patch(len(gb.convolutions)-1)
		print "Image input sizes:", sizes[0], "\tLabel input size:", sizes[1], "\tReceptive field:", r_field

		# Create architecture
		if load != '1':
			# Create placeholders
			x = make_placeholders(sizes, 'x')
			y = tf.placeholder(tf.float32, shape=[None, None, None, None], name = "y")
			weights = tf.placeholder(tf.float32, shape=[None, None, None, None], name = "weights")
			x.append(y)
			x.append(weights)

			# Create the pathways
			outputs = [[] for _ in range(gb.num_paths)]
			for pathway in range(gb.num_paths):
				path = create_path(x[pathway], str(pathway))

				# Upsample layers
				if pathway > 0:
					factor = gb.downsample[pathway-1]
					name = "deconv_" + str(pathway)
					path = deconv_layer(path, factor, name)

					cvs = len(gb.convolutions) - 1
					dif = ((sizes[0][pathway] - 2*cvs)*factor % sizes[1])%2
					path = tf.slice(path, [0, dif, dif, dif, 0], tf.shape(outputs[0]))
					path.set_shape(outputs[0].get_shape().as_list())

				outputs[pathway] = path

			# Concatenate the outputs
			concatenated = tf.concat(outputs, axis=-1) 

			# Dense layers
			fc_out = dense_block(concatenated)

			# Classification layer
			last_conv = last_convolution(fc_out, gb.num_classes, "final_conv")
			classification = tf.nn.softmax(logits=last_conv, name="softmax")
			prediction = tf.argmax(classification, axis=-1)
			tf.summary.scalar("Nonzero", tf.count_nonzero(prediction))

			# Loss
			y_temp = tf.cast(y, tf.int32)
			tf.summary.scalar("Nonzero_label", tf.count_nonzero(y_temp))
			loss_temp = tf.losses.sparse_softmax_cross_entropy(y_temp, last_conv, weights=weights)
			loss = tf.reduce_mean(loss_temp)
			tf.summary.scalar("loss", loss)

			learning_rate = tf.Variable(gb.learning_rate, name = "learning_rate")
			optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, name="optimizer")
			# optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.6, epsilon=10e-4).minimize(loss, name="optimizer")
			print "----- Architecture Created -----"

			accuracy = calculate_accuracy(prediction, y)
			tf.summary.scalar("accuracy", accuracy)

		with tf.Session() as sess: 
			init = (tf.global_variables_initializer(), tf.local_variables_initializer())
			last = 0 # Epochs until now

			# Restore model (if asked)
			if load == '1':
				ckpt = tf.train.get_checkpoint_state(gb.LOGDIR)
				loadpath = ckpt.model_checkpoint_path
				saver2 = tf.train.import_meta_graph(loadpath + '.meta')
				graph = tf.get_default_graph()  
				saver2.restore(sess, loadpath)

				x = tensor_names(mode)
				loss = graph.get_tensor_by_name("Mean:0")
				optimizer = graph.get_operation_by_name("optimizer")
				accuracy = graph.get_tensor_by_name("truediv:0")
				learning_rate = graph.get_tensor_by_name("learning_rate:0")

				index = loadpath.find('ckpt-')+5
				last = int(loadpath[index:])
				print "------------ Model", last, "restored ------------"

			saver = tf.train.Saver()
			summ = tf.summary.merge_all()
			writer = tf.summary.FileWriter(gb.LOGDIR + hparam)
			config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()      
			sess.run(init)

			writer.add_graph(sess.graph)
			image_list, label_list = get_list(mode)
			num_it = (gb.iterations*gb.patches_per_patient)/gb.batch_size
			count = last*gb.sub_epochs*num_it
			num_subepoch = last*gb.sub_epochs
			epochs_to_train = gb.num_epochs-last

			lista = range(gb.num_cases)

			for epoch in range(epochs_to_train):
				if last in gb.lr_decay:
					times = gb.lr_decay.index(last) + 1
					new_lr = gb.learning_rate/(2**times)
					reduce_lr = tf.assign(learning_rate, new_lr)
					sess.run(reduce_lr)
					print "New learning rate:", sess.run(learning_rate)

				cases = random.sample(lista, gb.iterations)
				for subepoch in range(gb.sub_epochs):
					num_subepoch += 1
					data_subepoch = input_pipeline(itemgetter(*cases)(image_list), itemgetter(*cases)(label_list), sizes, num_subepoch)

					acc_total = 0
					cost_total = 0
					print "----- Starting Epoch", last+1, "- Sub-epoch", subepoch+1, "-----"
					for iteration in range(num_it):
						data = [[] for _ in range(gb.num_paths+2)]
						for i in range(len(data_subepoch)):
							data[i] = np.squeeze(data_subepoch[i][:gb.batch_size], 0)
							del data_subepoch[i][:gb.batch_size]

						summ2, cost, acc, _ = sess.run([summ, loss, accuracy, optimizer], 
							 feed_dict={i: np.squeeze(d) for i, d in zip(x, data)})

						acc_total += acc
						cost_total += cost
						count += 1
						writer.add_summary(summ2, count)
						print "Iteration:", iteration+1, "\tloss:", cost, "\taccuracy:", acc
					print "Metrics sub-epoch:", subepoch+1, "loss:", cost_total/num_it, "accuracy:", acc_total/num_it
				last += 1
				saver.save(sess, os.path.join(gb.LOGDIR, "model.ckpt"), last)
				print "Model", last, "saved."

			print('Done training -- epoch limit reached')
			saver.save(sess, os.path.join(gb.LOGDIR, "model.ckpt"), last)
			sess.close()
			print "\n"+"Training complete!"

def test(mode):
	with tf.Graph().as_default():  
		# Set variables
		hparam = make_hparam_string(False, False)
		sizes = test_sizes_patches()
		print "Image input sizes:", sizes[0], "\tLabel input size:", sizes[1]

		testpath = '/media/user_home1/ladaza/Docker/data'
		outpath = '/media/user_home1/ladaza/Docker/data/results/'
		patient_list = os.listdir(testpath)
		patient_list.remove('results')


		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(gb.LOGDIR)
			loadpath = ckpt.model_checkpoint_path
			pdb.set_trace()
			model_saver = tf.train.import_meta_graph(loadpath + '.meta')
			graph = tf.get_default_graph()
			model_saver.restore(sess, loadpath)
			print "----- Model restored:", loadpath, "-----"
			x = tensor_names(test)
			output = graph.get_tensor_by_name("ArgMax:0") 
			min_diff = sizes[1]

			for num in patient_list:
				modalities = os.listdir(testpath + '/' + num)

				image = []
				for channel in range(gb.num_ch):
					temp = read_image(testpath + '/' + num + '/' + modalities[channel],0)
					# NORMALIZATION
					temp = (temp-np.mean(temp))/np.std(temp)
					image.append(np.pad(temp, min_diff, 'constant', constant_values=(0)))
				
				original = temp.shape
				patches = [int(math.ceil(p/float(sizes[1]))) for p in original]
				combinations = voxels_test(sizes[1], min_diff, original)
				data_test = input_test(image, sizes, patches, combinations)

				prob_out = []
				count = 0
				for iter2 in range(patches[0]):
					temporal = []
					for iter1 in range(patches[-1]):
						data = [[] for _ in range(gb.num_paths+1)]
						for i in range(len(data_test)):
							data[i] = np.squeeze(data_test[i][:patches[-1]], 0)
							del data_test[i][:patches[-1]]

						pred1 = sess.run([output], feed_dict={i: d for i, d in zip(x, data)})
						pred2 = np.split(pred1, patches[-1])
						pred = [np.squeeze(d) for d in pred2]

						temporal.append(np.concatenate(pred, axis=2))
					im_out.append(np.concatenate(temporal, axis=1))
				prediction = np.concatenate(im_out, axis=0)

				FinalPred = prediction[:original[0],:original[1],:original[2]]
				save_predictions(FinalPred,testpath+'/'+num+'/'+modalities[0], outpath+num+'.nii.gz')

				print "Prediction of patient", number, "saved."
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
