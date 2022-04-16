import torch

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
    return lambda pred_dict, target_dict: negative_SDR_dict(pred_dict,target_dict)

def negative_SDR_single(pred, target):
    assert torch.equal(pred.shape,target.shape)
    diff = target-pred
    diff = torch.mul(diff,diff)
    numerator = torch.sum(diff)
    
    denominator = torch.sum(torch.mul(target,target))
    
    logarithm = torch.log(numerator/denominator) / torch.log(torch.tensor(10.0))
    return torch.tensor(10.0) * logarithm

def negative_SDR_dict(pred_dict, target_dict):
    sdr_sum = torch.tensor(0.0)
    sdr_element_count = 0
    
    for i in pred_dict:
        sdr_element_count += 1
        sdr_sum += negative_SDR_single(pred_dict[i],target_dict[i])

    return sdr_sum / torch.tensor(sdr_element_count)