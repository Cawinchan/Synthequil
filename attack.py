import argparse
from datetime import datetime
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from alive_progress import alive_bar
from torch.utils.data import DataLoader

from dataset import DemixingAudioDataset
from utils import *


def main(dataset_dir: str, log_dir: str, custom_test_dir: Optional[str],
         test_output_dir: str,
         model: Optional[str]):
    # Get device to load samples and model to
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Get input directory, test model path
    if dataset_dir is None and custom_test_dir is None:
        raise Exception(
            "Error: no directory specified for testing, please use either --dataset-dir or --custom-test-dir")
    if model is None:
        raise Exception("Error: no test model specified, please use --model for this")

    # Set folder from which to get audio inputs
    input_dir = custom_test_dir if custom_test_dir is not None else os.path.join(dataset_dir, "test")

    # Initialize model and optimizer
    print("Using model at {}".format(model))

    # Load model and optimizer, as well as hyperparameters, and update hyperparameters and epoch count
    audio_model, optimizer, hyperparam_dict = load_model_and_optimizer(device, model)
    block_count = hyperparam_dict['block_count']
    learning_rate = hyperparam_dict['learning_rate']
    dropout_proba = hyperparam_dict['dropout_proba']
    dropout = hyperparam_dict['dropout']
    scale_pow = hyperparam_dict['scale_pow']
    elapsed_epoch = hyperparam_dict['epoch']

    # Get string indicating start time of training/testing
    current_datetime = datetime.today()
    time_str = "{}_{}_{}_{}_{}".format(current_datetime.year,
                                       current_datetime.month, current_datetime.day, current_datetime.hour,
                                       current_datetime.minute)

    # Get model configuration identifier
    model_id = "b{}_lr{}_d{}_scl{}".format(block_count, learning_rate, dropout_proba if dropout else "N", scale_pow)

    check_make_dir(test_output_dir)
    model_output_dir = os.path.join(test_output_dir, "{}-epoch_{}-{}".format(model_id, elapsed_epoch, time_str))
    check_make_dir(model_output_dir)
    print("Storing test outputs at {}".format(model_output_dir))

    # Define maximum chunk size in terms of number of samples, given sampling rate and max duration
    chunk_size = calculate_chunk_size(CLIP_TIME * SAMPLING_RATE, block_count)
    print("Input to be divided into chunks of {} samples".format(chunk_size))

    # Get Dataset object
    audio_dataset = DemixingAudioDataset(input_dir, chunk_size)

    # Split datasets only if training
    test_dataset = audio_dataset

    # Get first 5 for attack purposes
    test_dataloader = DataLoader(torch.utils.data.Subset(test_dataset, range(5)), shuffle=False)

    # Define loss criterion
    criterion = negative_SDR()

    # Validation/Testing
    # Track total loss and number of chunks processed

    epsilons = [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    total_loss = {eta: 0 for eta in epsilons}
    chunk_count = 0

    # Again, initialize the progress bar
    with alive_bar(len(test_dataloader)) as bar:

        for idx, i in enumerate(test_dataloader):

            # Set model to evaluation mode
            audio_model.eval()

            # Keep list of mixture chunks and predicted/target component chunks, if testing
            attack_chunks = {eps: list() for eps in epsilons}
            pred_chunks = {eps: {i: list() for i in INSTRUMENTS} for eps in epsilons}
            target_chunks = {i: list() for i in INSTRUMENTS}

            # Iterate through each chunk
            for segment_idx in range(len(i[0])):
                # Skip chunks without proper size
                input_data = i[0][segment_idx].detach()
                if input_data.shape[-1] < chunk_size:
                    continue

                # Move chunk to GPU or alternative
                original_input_data = input_data.to(device)

                for instr in INSTRUMENTS:
                    target = i[1][instr][segment_idx]
                    target_chunks[instr].append(target.detach())

                for eps in epsilons:

                    input_data = destroy(original_input_data, chunk_size, criterion, audio_model, eps, 1).to(device)

                    # Add mixture chunk to list if testing
                    attack_chunks[eps].append(input_data.cpu().detach())

                    for instr in INSTRUMENTS:
                        target = i[1][instr][segment_idx].detach()

                        pred = audio_model(input_data, instr)
                        loss = criterion(pred, target.to(pred.device))
                        # Save prediction if testing
                        pred_chunks[eps][instr].append(pred.cpu().detach())
                        total_loss[eps] += loss.item()

                chunk_count += len(INSTRUMENTS)

            # Save audio files if testing
            save_attacks(attack_chunks, pred_chunks, target_chunks,
                         os.path.join(model_output_dir, str(idx)))
            bar()
        for eps in epsilons:
            print(f"eps = {eps}: loss = {total_loss[eps]/chunk_count}")


def destroy(data: torch.Tensor,
            chunk_size: int,
            loss_fn,
            model: nn.Module, eps: float,
            iterations: int):
    noise = {i: torch.randn(2, chunk_size).to(data.get_device())/2 for i in INSTRUMENTS}
    for _ in range(iterations):
        data.requires_grad = True
        data.grad = None
        for inst in INSTRUMENTS:
            target = noise[inst]
            predicted = model(data, inst)
            loss = loss_fn(target, predicted)
            loss.backward()
        data.requires_grad = False
        if data.grad is not None:
            # stop pycharm from complaining that data.grad is set to None,
            # also prevent weird errors if there are no instruments (lol)
            data -= torch.nan_to_num(data.grad) * eps
            data = data.detach().to(data.get_device())
    return data


if __name__ == "__main__":
    # Get argument parser
    parser = argparse.ArgumentParser(description="Training/testing program for Audio Demixing")
    parser.add_argument("--dataset-dir", metavar="[dataset root dir]",
                        help="Root directory for dataset, containing train and test folders; " +
                             "ignored if custom input is specified for test mode")
    parser.add_argument("--log-dir", metavar="[root log dir]",
                        help="Root directory to store training/testing logs (default: ./logs)", default="./logs")
    parser.add_argument("--custom-test-dir", metavar="[custom test input folder path]",
                        help="Custom input folder for testing")
    parser.add_argument("--test-output-dir", metavar="[output root dir to store test outputs]",
                        help="Root directory to store test outputs; ignored if training (default: ./test_output)",
                        default="./test_output")
    parser.add_argument("--model", metavar="[path to model checkpoint]",
                        help="File path to model checkpoint for testing or continuing training")

    # Parse arguments and call main function
    args = parser.parse_args()
    main(args.dataset_dir, args.log_dir, args.custom_test_dir,
         args.test_output_dir, args.model)
