# List of instruments to segment into
INSTRUMENTS = ("bass", "drums", "vocals", "other")

# Portion of training dataset to be used for training
TRAIN_SPLIT = 0.8

# Size of kernel for convolution layers
KERNEL_SIZE = 4

# Stride size
STRIDE = 2

# Sampling rate of input music files
SAMPLING_RATE = 44100

# Maximum duration of audio chunks
CLIP_TIME = 15

# Depth of downsampling/upsampling preshortcut and postshortcut block segments
SAMPLE_BLOCK_DEPTH = 1

# Number of 1x1 convolution layers in bottleneck section of model
BOTTLENECK_DEPTH = 1

# SGD momentum
SGD_MOMENTUM = 0.9