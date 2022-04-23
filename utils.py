import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from unet_model import UNet
import math
from constants import *

# Save model, optimizer and hyperparameters
def save_model_and_optimizer(model: nn.DataParallel, optimizer: torch.optim, block_count: int,learning_rate: float, dropout_proba: float,
	dropout: bool, scale_pow: float, epoch: int, path: str):
		
		torch.save({
				'block_count': block_count,
				'learning_rate': learning_rate,
				'dropout_proba': dropout_proba,
				'dropout': dropout,
				'scale_pow': scale_pow,
				'epoch': epoch,   
				'model_state_dict': model.module.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()
		},path)

# Load model and optimizer, as well as hyperparameters
def load_model_and_optimizer(device: torch.device, path: str):
		loaded_data = torch.load(path)
		model, optimizer = generate_model_and_optimizer(loaded_data['block_count'],loaded_data['dropout'],
			loaded_data['dropout_proba'],loaded_data['scale_pow'],loaded_data['learning_rate'],device)
		model.module.load_state_dict(loaded_data['model_state_dict'])
		optimizer.load_state_dict(loaded_data['optimizer_state_dict'])
		del loaded_data['model_state_dict']
		del loaded_data['optimizer_state_dict']
		return model, optimizer, loaded_data

# Return loss function to calculate negative SDR of prediction
def negative_SDR():
		return lambda pred, target: negative_SDR_single(pred,target)

# Calculate negative SDR loss
def negative_SDR_single(pred: torch.Tensor, target: torch.Tensor):
		diff = target-pred
		diff = torch.mul(diff,diff)
		numerator = torch.sum(diff)
		
		denominator = torch.sum(torch.mul(target,target))
		
		logarithm = torch.log(numerator/denominator) / math.log(10)
		output = 10 * logarithm
		return output

# Calculate maximum size of chunk processable without
# tensor size mismatch in model, given maximum allowable samples in a given chunk
def calculate_chunk_size(maximum_chunk_length: int, block_count: int):

		# Get number of samples in bottleneck layer given maximum chunk length
		downsampled_length = maximum_chunk_length
		for i in range(2*SAMPLE_BLOCK_DEPTH*block_count):
				downsampled_length = 1 + ((downsampled_length-KERNEL_SIZE) // STRIDE)
		
		# Upsample number of samples
		upsampled_length = downsampled_length
		# (Yes, we can definitely make the next 2 lines O(1))
		for i in range(2*SAMPLE_BLOCK_DEPTH*block_count):
				upsampled_length = (upsampled_length-1)*STRIDE + KERNEL_SIZE
		return upsampled_length

# Check if dir exists, and if not, create it
def check_make_dir(dir: str):
	if not os.path.isdir(dir):
		os.makedirs(dir)

# Create the model and optimizer
def generate_model_and_optimizer(block_count: int, dropout: bool, dropout_proba: float, scale_pow: float,
	learning_rate: float, device: torch.device):
	feature_count_list = [2] + [16*(2**i) for i in range(block_count)]
	audio_model = nn.DataParallel(UNet(feature_count_list,KERNEL_SIZE,"leaky_relu",INSTRUMENTS,
		sample_block_depth=SAMPLE_BLOCK_DEPTH, bottleneck_depth=BOTTLENECK_DEPTH,dropout=dropout,dropout_proba=dropout_proba,
		scale_pow=scale_pow).to(device))
	return audio_model, torch.optim.Adam(audio_model.module.parameters(),learning_rate)

# Save test outputs with actual outputs
def save_outputs(mixture_chunks_list: list, pred_waveform_list_dict: dict,
	target_waveform_list_dict: dict, path: str):
	# Create directory
	check_make_dir(path)

	try:
		# Save original mixture wav
		torchaudio.save(os.path.join(path,"mixture.wav"), torch.cat(mixture_chunks_list,dim=-1).reshape((2,-1)),SAMPLING_RATE)

		# Save separated components (predicted and actual)
		for i in INSTRUMENTS:
			torchaudio.save(os.path.join(path,"{}_pred.wav".format(i)), torch.cat(pred_waveform_list_dict[i],dim=-1).reshape((2,-1)),SAMPLING_RATE,format="wav")
			torchaudio.save(os.path.join(path,"{}.wav".format(i)), torch.cat(target_waveform_list_dict[i],dim=-1).reshape((2,-1)),SAMPLING_RATE,format="wav")
	except Exception as e:
		raise Exception("Error: \"{}\".\nYou may have run out of space.".format(e))

	return
