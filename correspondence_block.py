import os
import cv2
import torch
import numpy as np
import unet_model as UNET
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

from dataset_classes import LineMODDataset


def train_correspondence_block(root_dir, train_eval_dir, classes, epochs=10, batch_size=4, \
                                out_path_and_name=None, corr_transfer=None):

    train_data = LineMODDataset(root_dir, train_eval_dir, classes=classes,
                                transform=transforms.Compose([transforms.ToTensor(),
                                transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)]))

    num_workers = 0
    valid_size = 0.2
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)

    # architecture for correspondence block - 13 objects + backgound = 14 channels for ID masks
    correspondence_block = UNET.UNet(n_channels=3, out_channels_id=14, bilinear=True)

    if corr_transfer:
        print("Initializing correspondence block from: %s" % corr_transfer)
        correspondence_block.load_state_dict(torch.load(corr_transfer, map_location=torch.device('cpu')))

    correspondence_block.cuda()

    # custom loss function and optimizer
    weight_classes = True
    if weight_classes:
        # Using weighted version for class mask as mentioned in the paper
        # However, not sure what the weighting is, so taking a guess
        # Note we don't need to normalize when using the default 'reduction' arg
        class_weights = np.ones(len(classes)+1) # +1 for background
        class_weights[0] = 0.5
        criterion_id = nn.CrossEntropyLoss(torch.tensor(class_weights, dtype=torch.float32).cuda())
    else:
        criterion_id = nn.CrossEntropyLoss()

    # specify optimizer
    optimizer = optim.Adam(correspondence_block.parameters(), lr=3e-4, weight_decay=3e-5)

    # training loop

    # number of epochs to train the model
    n_epochs = epochs

    # track change in validation loss
    # TODO - if transfer learning, don't assume infinity here, run validation once first
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs+1):
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        train_idmask_loss = 0.0
        valid_idmask_loss = 0.0

        print("------ Epoch ", epoch, " ---------")
        print("Training...")

        ###################
        # train the model #
        ###################
        batch_cnt = 0
        correspondence_block.train()
        for _, image, idmask, umask, vmask in train_loader:
            if batch_cnt % 100 == 0:
                print("Batch %i/%i finished!" % (batch_cnt, len(train_idx)/batch_size))

            # move tensors to GPU if CUDA is available
            image, idmask = image.cuda(), idmask.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            idmask_pred = correspondence_block(image)
            # calculate the batch loss
            loss_id = criterion_id(idmask_pred, idmask)
            total_loss = loss_id
            # backward pass: compute gradient of the loss with respect to model parameters
            total_loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_idmask_loss += loss_id.item()
            train_loss += total_loss.item()
            batch_cnt += 1
        ######################
        # validate the model #
        ######################
        print("Validating...")
        correspondence_block.eval()
        batch_cnt = 0
        with torch.no_grad(): # This is critical to limit GPU memory use
            for _, image, idmask, umask, vmask in valid_loader:
                if batch_cnt % 100 == 0:
                    print("Batch %i/%i finished!" % (batch_cnt, len(valid_idx)/batch_size))
                # move tensors to GPU if CUDA is available
                image, idmask = image.cuda(), idmask.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                idmask_pred = correspondence_block(image)
                # calculate the batch loss
                loss_id = criterion_id(idmask_pred, idmask)
                total_loss = loss_id
                # update average validation loss
                valid_idmask_loss += loss_id.item()
                valid_loss += total_loss.item()
                batch_cnt += 1

        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        train_idmask_loss = train_idmask_loss/len(train_loader.sampler)

        valid_loss = valid_loss/len(valid_loader.sampler)
        valid_idmask_loss = valid_idmask_loss/len(valid_loader.sampler)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        print('Train IDMask loss: %.6f' % (train_idmask_loss))
        print('Valid IDMask loss: %.6f' % (valid_idmask_loss))

        # TODO - monitor for train/val divergence and stop

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))

            if not out_path_and_name:
                correspondence_block_filename = os.path.join(train_eval_dir, 'correspondence_block.pt')
            else:
                correspondence_block_filename = out_path_and_name

            torch.save(correspondence_block.state_dict(), correspondence_block_filename)
            valid_loss_min = valid_loss
