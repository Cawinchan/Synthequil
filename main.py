import argparse
import chunk
import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import DemixingAudioDataset
from unet_model import UNet
from torch.utils.data import DataLoader, random_split
from utils import save_model, load_model, negative_SDR, calculate_chunk_size
from alive_progress import alive_bar

INSTRUMENTS = ("bass", "drums", "vocals", "other")
TRAIN_SPLIT = 0.8
KERNEL_SIZE = 4
FEATURE_COUNT_LIST = [2] + [16*(2**i) for i in range(6)]
SAMPLING_RATE = 44100
CLIP_TIME = 15

def main(dataset_dir, test, custom_test_dir, train_checkpoint_dir, model, epoch_count):

    # Define chunk size
    chunk_size = calculate_chunk_size(CLIP_TIME*SAMPLING_RATE,1,len(FEATURE_COUNT_LIST),KERNEL_SIZE)
    print("Input to be divided into chunks of {} samples".format(chunk_size))

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
    audio_dataset = DemixingAudioDataset(input_dir,chunk_size)
    train_len = int(0.8*len(audio_dataset))
    train_dataset, test_dataset = random_split(audio_dataset,[train_len,len(audio_dataset)-train_len],
        generator=torch.Generator().manual_seed(100)) if is_train else (None, audio_dataset)

    # Get DataLoader objects
    train_dataloader = DataLoader(train_dataset,shuffle=True)
    test_dataloader = DataLoader(test_dataset,shuffle=True)

    # Get device to load samples and model to
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define model and optimizer
    audio_model = nn.DataParallel(UNet(FEATURE_COUNT_LIST,KERNEL_SIZE,"leaky_relu",INSTRUMENTS))
    optimizer = optim.Adam(audio_model.parameters(),0.01)

    # Define loss criterion
    criterion = negative_SDR()
    
    # Load model and epoch count if needed
    epoch = 0
    if not model==None:
        epoch = load_model(audio_model,optimizer,model)
    
    for current_epoch in range(epoch if is_train else 0,epoch_count if is_train else 1):
        
        print("Epoch {}:".format(current_epoch+1))

        if is_train:

            total_loss = 0
            item_count = 0

            with alive_bar(len(train_dataloader)) as bar:
                for i in train_dataloader:
                
                    audio_model.train()

                    for segment_idx in range(len(i[0])):
                    
                        input_data = i[0][segment_idx].to(device);
                        if (input_data.shape[-1]<chunk_size): continue
                    

                        for instr in INSTRUMENTS:
                    
                            optimizer.zero_grad()

                            target = i[1][instr][segment_idx].to(device)

                            pred = audio_model(input_data,instr)
                            loss = criterion(pred,target)
                            loss.backward()
                            optimizer.step()

                            if True in torch.isnan(loss):
                                print(input_data,pred,target,loss)
                                raise Exception("Nan value found for loss, please diagnose")

                            total_loss += loss.item()
                            item_count += 1
                    bar()
            
            avg_loss = total_loss / item_count
            print("\tAverage loss during training: {}".format(avg_loss))
            print("\tSaving model...")
            save_model(audio_model,optimizer,current_epoch+1,os.path.join(train_checkpoint_dir,"model_" + str(current_epoch+1)))
        
        total_loss = 0
        item_count = 0

        with alive_bar(len(test_dataloader)) as bar:
            
            for i in test_dataloader:
            
                audio_model.eval()

                with torch.no_grad():
                
                    for segment_idx in range(len(i[0])):

                        input_data = i[0][segment_idx].to(device);
                        if (input_data.shape[-1]<chunk_size): continue

                        for instr in INSTRUMENTS:

                            target = i[1][instr][segment_idx].to(device)

                            pred = audio_model(input_data,instr)
                            loss = criterion(pred,target)

                            if True in torch.isnan(loss):
                                print(input_data,pred,target)
                                raise Exception("Nan value found for loss, please diagnose")

                            total_loss += loss.item()
                            item_count += 1
                bar()
        
        avg_loss = total_loss / item_count
        print("\tAverage loss during validation/test: {}".format(avg_loss))

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