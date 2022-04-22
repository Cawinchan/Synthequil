import argparse
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import DemixingAudioDataset
from torch.utils.data import DataLoader, random_split
from utils import check_make_dir, generate_model_and_optimizer, load_model_and_optimizer, save_model_and_optimizer, load_model_and_optimizer, negative_SDR, calculate_chunk_size, save_outputs
from alive_progress import alive_bar
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from constants import *

def main(dataset_dir: str, log_dir: str, train: bool, custom_test_dir: Optional[str], train_checkpoint_dir: str, test_output_dir: str,
    model: Optional[str], epoch_count: int, learning_rate: float, block_count: int, dropout: bool, dropout_proba: float, scale_pow: float):

    # Get device to load samples and model to
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Get input directory, test model path
    if train:
        if dataset_dir==None:
            raise Exception("Error: no dataset specified for training, please use --dataset-dir for this")
    else:
        if dataset_dir==None and custom_test_dir==None:
            raise Exception("Error: no directory specified for testing, please use either --dataset-dir or --custom-test-dir")
        if model==None:
            raise Exception("Error: no test model specified, please use --model for this")

    # Set folder from which to get audio inputs
    input_dir = None
    if train:
        input_dir = os.path.join(dataset_dir,"train")
    else:
        input_dir = custom_test_dir if custom_test_dir!=None else os.path.join(dataset_dir,"test")

    # Initialize number of elapsed epochs
    elapsed_epoch = 0

    # Initialize model and optimizer
    audio_model = None
    optimizer = None
    if model:
        print("Using model at {}".format(model))

        # Load model and optimizer, as well as hyperparameters, and update hyperparameters and epoch count
        audio_model, optimizer, hyperparam_dict = load_model_and_optimizer(device,model)
        block_count = hyperparam_dict['block_count']
        learning_rate = hyperparam_dict['learning_rate']
        dropout_proba = hyperparam_dict['dropout_proba']
        dropout = hyperparam_dict['dropout']
        scale_pow = hyperparam_dict['scale_pow']
        elapsed_epoch = hyperparam_dict['epoch']
    else:

        # Enforce hyperparameter constraints
        if (block_count<1): raise Exception("Error: block count must be at least 1")
        if (learning_rate<=0): raise Exception("Error: learning rate must be above 0")
        if (dropout and (dropout<0 or dropout>1)): raise Exception("Error: dropout must be 0-1 inclusive")
        if (scale_pow<=0): raise Exception("Error: input scaling power must be above 0")
        
        # Create new model and optimizer from scratch
        audio_model, optimizer = generate_model_and_optimizer(block_count,dropout,dropout_proba,scale_pow,learning_rate,device)

    # Get string indicating start time of training/testing
    current_datetime = datetime.today()
    time_str = "{}_{}_{}_{}_{}".format(current_datetime.year,
        current_datetime.month,current_datetime.day,current_datetime.hour,current_datetime.minute)

    # Get model configuration identifier
    model_id = "b{}_lr{}_d{}_scl{}".format(block_count, learning_rate, dropout_proba if dropout else "N", scale_pow)

    # Check and create directories for log root folder and for model-specific folder
    check_make_dir(log_dir)
    model_log_dir = os.path.join(log_dir,"train" if train else "test")
    check_make_dir(model_log_dir)
    model_log_dir = os.path.join(model_log_dir,model_id)
    check_make_dir(model_log_dir)
    model_log_dir = os.path.join(model_log_dir,time_str)
    check_make_dir(model_log_dir)
    print("Logs to be stored at {}".format(model_log_dir))

    # Get Tensorboard SummaryWriter
    writer = SummaryWriter(model_log_dir)

    # Check and create directories for checkpoint root folder and for model-specific folder, if training
    model_checkpoint_dir = None
    if train:
        check_make_dir(train_checkpoint_dir)
        model_checkpoint_dir = os.path.join(train_checkpoint_dir,model_id)
        check_make_dir(model_checkpoint_dir)
        print("Model checkpoints to be saved at {}".format(model_checkpoint_dir))

    # Check and create directories for test outputs, if testing
    model_output_dir = None
    if not train:
        check_make_dir(test_output_dir)
        model_output_dir = os.path.join(test_output_dir,"{}-epoch_{}-{}".format(model_id,elapsed_epoch,time_str))
        check_make_dir(model_output_dir)
        print("Storing test outputs at {}".format(model_output_dir))

    # Define maximum chunk size in terms of number of samples, given sampling rate and max duration
    chunk_size = calculate_chunk_size(CLIP_TIME*SAMPLING_RATE,block_count)
    print("Input to be divided into chunks of {} samples".format(chunk_size))

    # Get Dataset object
    audio_dataset = DemixingAudioDataset(input_dir,chunk_size)

    # Split datasets only if training
    train_len = int(TRAIN_SPLIT*len(audio_dataset))
    train_dataset, test_dataset = random_split(audio_dataset,[train_len,len(audio_dataset)-train_len],
        generator=torch.Generator().manual_seed(100)) if train else (None, audio_dataset)

    # Get DataLoader objects
    train_dataloader = DataLoader(train_dataset,shuffle=True) if not train_dataset==None else None
    test_dataloader = DataLoader(test_dataset,shuffle=True)

    # Define loss criterion
    criterion = negative_SDR()
    
    # Iterate only once if testing, otherwise iterate until target number of epochs is reached
    for current_epoch in range(elapsed_epoch if train else 0, epoch_count if train else 1):
        
        print("Epoch {}:".format(current_epoch+1))

        if train:

            # Training
            # Track total loss and number of chunks processed
            total_loss = 0
            chunk_count = 0

            # Setup nice-looking progress bar for each epoch portion
            with alive_bar(len(train_dataloader)) as bar:
                
                for i in train_dataloader:

                    # Set model to training mode
                    audio_model.train()

                    # Iterate through each chunk for a given track
                    for segment_idx in range(len(i[0])):
                    
                        # Load input, but only if it has proper chunk size
                        input_data = i[0][segment_idx];
                        if (input_data.shape[-1]<chunk_size): continue
                        input_data = input_data.to(device)

                        # Iterate through each instrument for given chunk
                        for instr in INSTRUMENTS:
                    
                            # Clear gradients
                            optimizer.zero_grad()

                            # Get prediction target
                            target = i[1][instr][segment_idx].to(device)

                            # Get prediction and loss, then backpass and backpropagate
                            pred = audio_model(input_data,instr)
                            loss = criterion(pred,target)
                            loss.backward()
                            optimizer.step()

                            # Update loss and chunk counters
                            total_loss += loss.item()
                            chunk_count += 1
                    
                    # Update progress bar
                    bar()
            
            # Calculate average loss
            avg_loss = total_loss / chunk_count
            print("\tAverage loss during training: {}".format(avg_loss))
            print("\tSaving model...")

            # Save model and optimizer parameters
            save_model_and_optimizer(audio_model,optimizer,block_count,learning_rate,dropout_proba,dropout,scale_pow,current_epoch+1,
                os.path.join(model_checkpoint_dir,"epoch_{}-{}".format(current_epoch+1,time_str)))
            
            # Record loss
            writer.add_scalar('Average_Loss (Train)', avg_loss, current_epoch+1)
        
        # Validation/Testing
        # Track total loss and number of chunks processed
        total_loss = 0
        chunk_count = 0

        # Again, initialize the progress bar
        with alive_bar(len(test_dataloader)) as bar:
            
            for idx, i in enumerate(test_dataloader):
            
                # Set model to evaluation mode
                audio_model.eval()

                # Disable gradient tracking
                with torch.no_grad():

                    # Keep list of mixture chunks and predicted/target component chunks, if testing
                    mixture_chunk_list, pred_chunk_list_dict, target_chunk_list_dict = None, None, None
                    if not train:
                        mixture_chunk_list = []
                        pred_chunk_list_dict = {i:list() for i in INSTRUMENTS}
                        target_chunk_list_dict = {i:list() for i in INSTRUMENTS}
                
                    # Iterate through each chunk
                    for segment_idx in range(len(i[0])):

                        # Skip chunks without proper size
                        input_data = i[0][segment_idx]
                        if (input_data.shape[-1]<chunk_size): continue

                        # Add mixture chunk to list if testing
                        if not train: mixture_chunk_list.append(input_data.detach())

                        # Move chunk to GPU or alternative
                        input_data = input_data.to(device)

                        for instr in INSTRUMENTS:

                            target = i[1][instr][segment_idx]

                            # Save target if testing
                            if not train: target_chunk_list_dict[instr].append(target.detach())
                            target = target.to(device)

                            pred = audio_model(input_data,instr)
                            loss = criterion(pred,target)
                            # Save prediction if testing
                            if not train: pred_chunk_list_dict[instr].append(pred.cpu().detach())

                            total_loss += loss.item()
                            chunk_count += 1
                    
                    # Save audio files if testing
                    if not train: save_outputs(mixture_chunk_list,pred_chunk_list_dict,target_chunk_list_dict,os.path.join(model_output_dir,str(idx)))
                bar()
        
        avg_loss = total_loss / chunk_count
        print("\tAverage loss during validation/test: {}".format(avg_loss))

        writer.add_scalar('Average_Loss (Validation or Test)', avg_loss, current_epoch+1)

if __name__=="__main__":
    
    # Get argument parser
    parser = argparse.ArgumentParser(description="Training/testing program for Audio Demixing")
    parser.add_argument("--dataset-dir", metavar="[dataset root dir]", help="Root directory for dataset, containing train and test folders; ignored if custom input is specified for test mode")
    parser.add_argument("--log-dir", metavar="[root log dir]", help="Root directory to store training/testing logs (default: ./logs)", default="./logs")
    parser.add_argument("--test", help="Toggle test mode", action="store_true")
    parser.add_argument("--custom-test-dir", metavar="[custom test input folder path]", help="Custom input folder for testing")
    parser.add_argument("--train-checkpoint-dir", metavar="[root directory to store checkpoints]", help="Root directory to store checkpoints of model during training (default: ./checkpoints)", default="./checkpoints")
    parser.add_argument("--test-output-dir", metavar="[output root dir to store test outputs]", help="Root directory to store test outputs; ignored if training (default: ./test_output)", default="./test_output")
    parser.add_argument("--model", metavar="[path to model checkpoint]", help="File path to model checkpoint for testing or continuing training")
    parser.add_argument("--epoch-count", metavar="[number of epochs to train model]", help="Number of epochs by which to train model (default: 50)", default=50)
    parser.add_argument("--learning-rate", metavar="[learning rate]", help="Learning rate of model; ignored if loading model (default: 0.01)", default=0.01)
    parser.add_argument("--block-count", metavar="[number of downsampling/upsampling blocks]", help="Number of downsampling and of upsampling blocks in model; ignored if loading model (default: 1)", default=1)
    parser.add_argument("--dropout", help="Toggle dropout for training; ignored if loading model", action="store_true")
    parser.add_argument("--dropout-proba", metavar="[dropout probability]", help="Probability used for dropout layers; ignored if --dropout is not used as well (default: 0.2)", default=0.2)
    parser.add_argument("--scale-pow", metavar="[power to scale inputs by]", help="Exponent to scale input samples by, while preserving amplitude signs (default: 0.5)", default=0.5)

    # Parse arguments and call main function
    args = parser.parse_args()
    main(args.dataset_dir, args.log_dir, not args.test, args.custom_test_dir, args.train_checkpoint_dir, args.test_output_dir, args.model, int(args.epoch_count),
        float(args.learning_rate), int(args.block_count), args.dropout, float(args.dropout_proba), float(args.scale_pow))