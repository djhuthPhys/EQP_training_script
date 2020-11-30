import random
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import EQP_TasNet_v0_2 as eqp

from tqdm import tqdm

def load_eqp_data():
    """
    Loads the EQP experiment data from the 2017-2018 science run and returns the time data, filtered data, and
    unfiltered data as a vector. Excludes the first and last 1,000,000 points from Tukey windowing during filtering
    :return: time_data, raw_position, filtered_position
    """

    with open('EQP_2018_run.pkl', 'rb') as handler:
        raw_data = pkl.load(handler)

    with open('EQP_2018_run_filtered.pkl', 'rb') as handler:
        filtered_data = pkl.load(handler)

    time = raw_data[999999:-999999, 0]
    raw_position = (raw_data[999999:-999999, 1]-1024)/1720
    filtered_position = filtered_data[999999:-999999, 1]/1720

    del raw_data, filtered_data

    # Convert to torch tensors
    time = torch.tensor(time).float()
    raw_position = torch.tensor(raw_position, requires_grad=True).float()
    filtered_position = torch.tensor(filtered_position, requires_grad=True).float()

    # Reshape for training
    num_examples = raw_position.size()[0]
    raw_position = torch.reshape(raw_position, (num_examples, 1, 1))
    filtered_position = torch.reshape(filtered_position, (num_examples, 1, 1))

    return time, raw_position, filtered_position


def build_x_batch(X, indices, window_size):
    """
    Builds the batches for input data. Done to save memory allocation for large data sets
    :param X: input data
    :param indices: list of indices for the batch
    :param window_size: defines input window length
    :return: X_batch
    """

    X_batch = torch.empty((indices.size()[0], 1, window_size))
    i = 0
    for index in indices:
        X_batch[i, :, :] = torch.transpose(X[index - window_size:index, :, :], 0, 2)
        i += 1

    return X_batch


def train_model(model, criterion, optimizer, X, Y, X_val, Y_val, batch_size=16, window_size=16):
    """
    Function for training the model
    :param model: the model to train
    :param criterion: defines loss function for training
    :param optimizer: defines optimization algorithm
    :param X: input data for network, chunked
    :param Y: ground truth data, chunked
    :param X_val: input data for network evaluation, chunked
    :param Y_val: ground truth for network evaluation, chunked
    :param batch_size: defines batch size for training
    :param window_size: defines the number of prior data points the network sees for each prediction
    :return:
    """
    # Initialize variables
    permutation = torch.randperm(Y.size()[0]-window_size, requires_grad=False) + window_size # Need to ignore initial data points in a length of window_size
    num_batches = 0
    total_batches = 0
    running_loss= 0.0

    # Train on a batch
    for i in tqdm(range(0, Y.size()[0]-window_size, batch_size)):
        # Zero out parameter gradients
        optimizer.zero_grad()

        # Define batch
        if (i+window_size) <= Y.size()[0] - window_size:
            indices = permutation[i:i+batch_size]
        else:
            indices = permutation[i:Y.size()[0] - window_size]
        Y_batch = Y[indices, :, :]
        X_batch = build_x_batch(X, indices, window_size) # Build input batch matrix from X vector
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # Forward, backward, and parameter update steps
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        num_batches += 1
        total_batches += 1

        # Print metrics, evaluate model
        if (i/batch_size) % 10 == 9:
            print('Training loss = %.8f' % (running_loss/num_batches))
            wandb.log({'Training loss': running_loss/num_batches})
            running_loss = 0.0
            num_batches = 0
        if (i/batch_size) % 100 == 99:
            val_loss = evaluate_model(model, criterion, X_val, Y_val, batch_size, window_size)
            print('Validation loss = %.8f' % val_loss)
            wandb.log({'Validation loss': val_loss}, commit=False)
        break
        #
        # if total_batches == 1000:
        #     break


def evaluate_model(model, criterion, X_val, Y_val, batch_size=16, window_size=16):
    """
    Evaluates model with validation data, data order is conserved since model is not training
    :param model:
    :param criterion
    :param X_val:
    :param Y_val:
    :param batch_size:
    :param window_size:
    :return:
    """
    running_loss = 0.0
    num_batches = 0
    model.eval()
    outputs = torch.empty(Y_val.size()[0] - window_size)

    for i in range(0, Y_val.size()[0] - window_size, batch_size)[0:-1]:
        # Define batch
        indices = np.linspace(i,i+batch_size-1, batch_size, dtype=int) + window_size
        Y_batch = Y_val[indices, :, :].to(device)
        X_batch = build_x_batch(X_val, indices, window_size).to(device)

        with torch.no_grad():
            output = model(X_batch)
            loss = criterion(output, Y_batch)
            running_loss += float(loss.item())
            outputs[i:i + batch_size] = torch.squeeze(output)
        num_batches += 1

    model.train()
    plt.figure()
    plt.plot(np.linspace(0, len(outputs), len(outputs), dtype=int), outputs)
    plt.plot(np.linspace(0, Y_val.size()[0]-window_size-1, Y_val.size()[0]-window_size-1, dtype=int), torch.squeeze(Y_val[window_size:-1,:,:]).detach().numpy())
    plt.ylim((-1,1))
    plt.show()

    return running_loss/num_batches


hp_defaults = dict(depth_channel=128,
                   bottleneck_channel=512,
                   batch_size=128,
                   window_size=128,
                   num_blocks=5,
                   depths=7)

wandb.init(project='tasnet_adaptation_v0_1', config=hp_defaults, mode='disabled')
config = wandb.config

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Training on GPU')
else:
    device = torch.device('cpu')
    print('Training on CPU')


def main(depth_channels, bottleneck_channels, batch_size, window_size, num_blocks, depths):
    """
    Main function for training and evaluating TasNet adaptation
    :param depth_channels:
    :param bottleneck_channels:
    :param batch_size:
    :param window_size:
    :param num_blocks: defines the number of blocks in the separation module
    :param depths:
    :return:
    """
    torch.cuda.empty_cache()

    # Define model, loss function, and optimizer
    depths = tuple([depths]*num_blocks)
    net = eqp.TasNetModel(depth_channels, bottleneck_channels, window_size, depths).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    wandb.watch(net, log='gradients')

    time_data, raw, filtered = load_eqp_data()

    data_set_length = raw.size()[0]

    # Separate train, validation, and test sets
    X_train = raw[0:round(0.98 * data_set_length), :, :]
    X_validation = raw[round(0.98 * data_set_length):round(0.99 * data_set_length), :, :]
    # X_test = raw[round(0.99 * data_set_length):data_set_length,:,:]

    Y_train = filtered[0:round(0.98 * data_set_length), :, :]
    Y_validation = filtered[round(0.98 * data_set_length):round(0.99 * data_set_length), :, :]
    # Y_test = filtered[round(0.99 * data_set_length):data_set_length, :, :]

    del raw, filtered


    # Break data into chunks, randomize order, and train
    chunk_size = 1000
    val_chunk_size = 10000
    chunk_start = list(range(0, X_train.size()[0], chunk_size))
    val_chunk_start = list(range(0, X_validation.size()[0] - val_chunk_size, val_chunk_size))
    random.shuffle(chunk_start)

    chunk_num = 1
    for chunk in chunk_start:
        print('Starting chunk %i' % chunk_num)
        X_chunk = X_train[chunk:chunk + chunk_size, :, :]
        Y_chunk = Y_train[chunk:chunk + chunk_size, :, :]

        val_index = random.choice(val_chunk_start)
        X_val_chunk = X_validation[val_index:val_index + val_chunk_size, :, :]
        Y_val_chunk = Y_validation[val_index:val_index + val_chunk_size, :, :]

        train_model(net, criterion, optimizer, X_chunk, Y_chunk, X_val_chunk, Y_val_chunk,
                    batch_size=batch_size,
                    window_size=window_size)
        chunk_num += 1
        break


if __name__ == '__main__':
    main(depth_channels=config.depth_channel,
         bottleneck_channels=config.bottleneck_channel,
         batch_size=config.batch_size,
         window_size=config.window_size,
         num_blocks=config.num_blocks,
         depths=config.depths)

# if __name__ == '__main__':
#     main(depth_channels=hp_defaults['depth_channel'],
#          bottleneck_channels=hp_defaults['bottleneck_channel'],
#          batch_size=hp_defaults['batch_size'],
#          window_size=hp_defaults['window_size'],
#          num_blocks=hp_defaults['num_blocks'],
#          depths=hp_defaults['depths'])