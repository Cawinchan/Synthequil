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
    return lambda pred, target: negative_SDR_single(pred,target)

def negative_SDR_single(pred, target):
    diff = target-pred
    diff = torch.mul(diff,diff)
    numerator = torch.sum(diff)
    
    denominator = torch.sum(torch.mul(target,target))
    
    logarithm = torch.log(numerator/denominator) / 10
    return 10 * logarithm