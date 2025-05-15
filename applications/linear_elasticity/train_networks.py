# MIT License
# Copyright (c) 2025
#
# This is part of the dino_tutorial package
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# For additional questions contact Thomas O'Leary-Roseberry

import os, sys
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('../../')


from dinotorch_lite import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n_train', '--n_train', type=int, default=800, help="Number of training data")

args = parser.parse_args()

assert args.n_train <= 800 and args.n_train > 0

data_dir = 'data/pointwise/'

mq_data_dict = np.load(data_dir+'mq_data_reduced.npz')
J_data_dict = np.load(data_dir+'JstarPhi_data_reduced.npz')

rM = 100
rQ = 100

m_data = mq_data_dict['m_data'][:,:rM]
q_data = mq_data_dict['q_data'][:,:rQ]
J_data = J_data_dict['J_data'][:,:rQ,:rM]
n_data,dQ,dM = J_data.shape


m_train = torch.Tensor(m_data[:args.n_train])
q_train = torch.Tensor(q_data[:args.n_train])
J_train = torch.Tensor(J_data[:args.n_train])

m_test = torch.Tensor(m_data[-200:])
q_test = torch.Tensor(q_data[-200:])
J_test = torch.Tensor(J_data[-200:])


# Set up datasets and loaders
l2train = L2Dataset(m_train,q_train)
l2test = L2Dataset(m_test,q_test)
batch_size = 32

train_loader = DataLoader(l2train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(l2test, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


################################################################################
# L2 training
model = GenericDense(input_dim = dM,hidden_layer_dim = 2*dM,output_dim = dQ).to(device)

n_epochs = 100
loss_func = normalized_f_mse
lr_scheduler = None

optimizer = torch.optim.Adam(model.parameters())

network, history = l2_training(model,loss_func,train_loader, validation_loader,\
                     optimizer,lr_scheduler=lr_scheduler,n_epochs = n_epochs)

rel_error = evaluate_l2_error(model,validation_loader)

print('L2 relative error = ', rel_error)

torch.save(model.state_dict(), data_dir+'l2_model_'+str(args.n_train)+'.pth')



################################################################################
# DINO training

dino_model = GenericDense(input_dim = dM,hidden_layer_dim = 2*dM,output_dim = dQ).to(device)

n_epochs = 100
loss_func_l2 = normalized_f_mse
loss_func_jac = normalized_f_mse
lr_scheduler = None

# Set up datasets and loaders
dinotrain = DINODataset(m_train,q_train, J_train)
dinotest = DINODataset(m_test,q_test, J_test)
batch_size = 32

dino_train_loader = DataLoader(dinotrain,  batch_size=batch_size, shuffle=True)
dino_validation_loader = DataLoader(dinotest, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(dino_model.parameters())

network, history = h1_training(dino_model,loss_func_l2, loss_func_jac, dino_train_loader, dino_validation_loader,\
                             optimizer,lr_scheduler=lr_scheduler,n_epochs = n_epochs, verbose=True)

rel_error = evaluate_l2_error(dino_model,validation_loader)

print('L2 relative error = ', rel_error)

torch.save(dino_model.state_dict(), data_dir+'dino_model_'+str(args.n_train)+'.pth')


