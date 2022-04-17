from random import sample
import torch
import math

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
    
    logarithm = torch.log(numerator/denominator) / math.log(10)
    return 10 * logarithm

def calculate_chunk_size(original_length,sample_block_depth,feature_list_len,kernel_size):

    stride = kernel_size //2

    downsampled_length = original_length
    for i in range(2*sample_block_depth*(feature_list_len-1)):
        downsampled_length = 1 + ((downsampled_length-kernel_size) // stride)
    
    upsampled_length = downsampled_length
    for i in range(2*sample_block_depth*(feature_list_len-1)):
        upsampled_length = (upsampled_length-1)*stride + kernel_size

    
    return upsampled_length