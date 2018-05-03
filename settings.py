class gb(object):
    # Paths
	LOGDIR = '/media/SSD3/Densenext_3.0/TRAIN/train9/'
	outpath = '/media/SSD3/Densenext_3.0/TEST/val/'
	impath = '/media/SSD3/Densenext/Paths/Train/train'
	# testpath = '/media/SSD3/Densenext/Paths/Train/train'
	testpath = '/media/SSD3/Densenext/Paths/Val/test'

	# Set variables
	learning_rate = 0.0001
	lr_decay = [3,6,9,12,15]
	num_epochs = 15
	sub_epochs = 20
	batch_size = 10
	factor = 100 # Number of patients per sub-epoch
	patches_per_patient = 20 # cuadrar para que sea divisible en num_classes (-1)
	num_classes = 5 # 2 o 5, binary or all
	num_ch = 4
	ends = ["Channels_flair.txt", "Channels_t1.txt", "Channels_t1c.txt", "Channels_t2.txt", "GtLabels.txt"] # Last one: annotations
	train_patches = 40 # Size Train patches
	test_patches = 20 # Size Test patches
	num_cases = 285 # Number of patients in the dataset

	# Architecture parameters
	cumulative = True
	num_paths = 2 # First will be normal, the others will receive a downsampled image
	downsample = [3] # Factors to downsample inputs
	fc_layers = 2 # Number of fully connected layers
	fc_drop = 0.5 # The probability that each element is kept
	total_blocks = 4
	cardinality = 2
	convolutions = [[8,4,8],[7,4,7],[6,4,6],[5,4,5],[4,4,4]] # initial convolution + layers [block]
	reduction = 1.0/8 # Compression of the number of feature maps in the transition layers
	bc_mode = False
	conv_drop = 0.98
