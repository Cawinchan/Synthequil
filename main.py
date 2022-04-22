import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import DemixingAudioDataset
from unet_model import UNet
from torch.utils.data import DataLoader, random_split
from utils import save_model, load_model, negative_SDR
import time
import math
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

INSTRUMENTS = ("bass", "drums", "vocals", "other")
TRAIN_SPLIT = 0.8
PATIENCE = 2

def main(dataset_dir, test, custom_test_dir, train_checkpoint_dir, model, epoch_count):
    patience_triggers = 0
    past_avg_loss = float(math.inf)

    # Get input directory, checkpoint directory, test model path
    input_dir = None
    if not test:
        if dataset_dir==None:
            raise Exception("Error: no dataset specified for training, please use --dataset-dir for this")
        input_dir = os.path.join(dataset_dir,"train")
        if not os.path.isdir(train_checkpoint_dir):
            os.mkdir(train_checkpoint_dir)
    else:
        if dataset_dir==None and custom_test_dir==None:
            raise Exception("Error: no directory specified for testing, please use either --dataset-dir or --custom-test-dir")
        input_dir = custom_test_dir if custom_test_dir!=None else os.path.join(dataset_dir,"test")
        if model==None:
            raise Exception("Error: no test model specified, please use --model for this")

    # Toggle train/test mode
    is_train = not test

    # Get Dataset object
    audio_dataset = DemixingAudioDataset(input_dir)
    train_len = int(0.8*len(audio_dataset))
    train_dataset, test_dataset = random_split(audio_dataset,[train_len,len(audio_dataset)-train_len],
    generator=torch.Generator().manual_seed(100)) if is_train else (None, audio_dataset)

    # Get DataLoader objects
    train_dataloader = DataLoader(train_dataset,shuffle=True)
    test_dataloader = DataLoader(test_dataset,shuffle=True)

    # Get device to load samples and model to
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define model and optimizer
    audio_model = nn.DataParallel(UNet([2**i for i in range(1,3)],5,"leaky_relu",INSTRUMENTS))
    # print(sum(p.numel() for p in audio_model.parameters() if p.requires_grad))
    # optimizer = optim.SGD(audio_model.parameters(),lr=0.1,momentum=0.9)
    optimizer = optim.Adam(audio_model.parameters(),lr=0.001) # optimiser recommended by wave-u-net paper

    # Define loss criterion
    criterion = negative_SDR()
    
    # Load model and epoch count if needed
    epoch = 0
    if not model==None:
        epoch = load_model(audio_model,optimizer,model)
    
    for current_epoch in range(epoch if is_train else 0,epoch_count if is_train else 1):
        
        print("Epoch {}:".format(current_epoch+1))
        start_time = time.time()


        if is_train:

            total_loss = 0

            for id, i in enumerate(train_dataloader):
                
                audio_model.train()

                input = i[0].to(device)
                target = i[1]

                for j in INSTRUMENTS:
                    optimizer.zero_grad()

                    pred = audio_model(input,j)

                    # writer.add_audio('{}_{}'.format(id,j), torch.reshape(pred,(1, -1)), current_epoch+1)
                    loss = criterion(pred,target[j].to(device))
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
            
            avg_loss = total_loss / len(train_dataloader) / len(INSTRUMENTS)
            print("/tAverage loss during training: {}".format(avg_loss))
            print("Saving model...")
            writer.add_scalar('Avergage_Loss/train', avg_loss, current_epoch+1)
            save_model(audio_model,optimizer,current_epoch+1,os.path.join(train_checkpoint_dir,"model_" + str(current_epoch+1)))
            end_time = time.time()
            print("Time taken", end_time-start_time)

            if avg_loss > past_avg_loss:
                patience_triggers += 1

            if patience_triggers >= PATIENCE:
                print('Early stopping!\nStart to test process.')
                break

            else:
                print('trigger times: {}'.format(patience_triggers))
                patience_triggers = 0

                past_avg_loss = avg_loss
        
        total_loss = 0
        with torch.no_grad():
            for i in test_dataloader:
                
                audio_model.eval()
                input = i[0].to(device)
                target = i[1]

                for j in INSTRUMENTS:
                    loss = criterion(audio_model(input,j),target[j].to(device))
                
                    total_loss += loss.item()
            
            avg_loss = total_loss / len(test_dataloader) / len(INSTRUMENTS)
            print("/tAverage loss during validation/test: {}".format(avg_loss))
            writer.add_scalar('Avergage_Loss/test', avg_loss, current_epoch+1)


if __name__=="__main__":
    
    # Get argument parser
    parser = argparse.ArgumentParser(description="Training/testing program for Audio Demixing")
    parser.add_argument("--dataset-dir", metavar="dataset root dir", help="Root directory for dataset, containing train and test folders; ignored if custom input is specified for test mode")
    parser.add_argument("--test", help="Toggle test mode", action="store_true")
    parser.add_argument("--custom-test-dir", metavar="custom test input folder path", help="Custom input folder for testing")
    parser.add_argument("--train-checkpoint_dir", metavar="directory to store checkpoints", help="Directory to store checkpoints of model during training (default: ./checkpoints)", default="./checkpoints")
    parser.add_argument("--model", metavar="path to model checkpoint", help="File path to model checkpoint for testing or continuing training")
    parser.add_argument("--epoch-count", metavar="number of epochs to train model", help="Number of epochs by which to train model (default: 50)", default=50)

    # Parse arguments and call main function
    args = parser.parse_args()
    main(args.dataset_dir, args.test, args.custom_test_dir, args.train_checkpoint_dir, args.model, int(args.epoch_count))