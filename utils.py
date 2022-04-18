import torch
import math
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import torchaudio.transforms as T

def save_model(model,optimizer,epoch,path):
    
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    },path)

def load_model(model,optimizer,path):

    loaded_data = torch.load(path)
    model.load_state_dict(loaded_data['model'])
    optimizer.load_state_dict(loaded_data['optimizer'])
    return loaded_data['epoch']

def negative_SDR():
    return lambda pred, target: negative_SDR_single(pred,target)

def negative_SDR_single(pred, target):
    diff = target-pred
    diff = torch.mul(diff,diff)
    numerator = torch.sum(diff)
    
    denominator = torch.sum(torch.mul(target,target))
    if denominator==0:
        denominator = 1
    
    logarithm = torch.log(numerator/denominator) / math.log(10)
    output = 10 * logarithm
    return output

def calculate_chunk_size(original_length,sample_block_depth,feature_list_len,kernel_size):

    stride = kernel_size //2

    downsampled_length = original_length
    for i in range(2*sample_block_depth*(feature_list_len-1)):
        downsampled_length = 1 + ((downsampled_length-kernel_size) // stride)
    
    upsampled_length = downsampled_length
    for i in range(2*sample_block_depth*(feature_list_len-1)):
        upsampled_length = (upsampled_length-1)*stride + kernel_size

    
    return upsampled_length

def print_stats(waveform, sample_rate=None, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  if sample_rate:
    print("Sample Rate:", sample_rate)
  print("Shape:", tuple(waveform.shape))
  print("Dtype:", waveform.dtype)
  print(f" - Max:     {waveform.max().item():6.3f}")
  print(f" - Min:     {waveform.min().item():6.3f}")
  print(f" - Mean:    {waveform.mean().item():6.3f}")
  print(f" - Std Dev: {waveform.std().item():6.3f}")
  print()
  print(waveform)
  print()

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=False)

def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")