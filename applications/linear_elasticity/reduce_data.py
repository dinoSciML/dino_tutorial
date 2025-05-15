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


import sys, os
import numpy as np

################################################################################
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', '--data_dir', type=str, default='data/pointwise/', help="What directory for all data to be split")
parser.add_argument('-input_basis', '--input_basis', type=str, default='as', help="What type of input basis? Choose from [kle] ")
parser.add_argument('-output_basis', '--output_basis', type=str, default='None', help="What type of input basis? Choose from [pod] ")

args = parser.parse_args()


# Load the data:
mq_file = 'mq_data.npz'
full_data = np.load(args.data_dir+mq_file)
m_data = full_data['m_data']
u_data = full_data['q_data']

# Load the projectors
if args.input_basis.lower() == 'kle':
	input_encoder_file = 'KLE_encoder.npy'
	input_decoder_file = 'KLE_decoder.npy'
else:
	input_encoder_file = 'AS_input_encoder.npy'
	input_decoder_file = 'AS_input_decoder.npy'

input_encoder = np.load(args.data_dir + input_encoder_file)
input_decoder = np.load(args.data_dir + input_decoder_file)

if args.output_basis.lower() == 'none':
	output_encoder = None
else:
	if args.output_basis.lower() == 'pod':
		output_encoder_file = 'POD/POD_encoder.npy'
		output_shift_file = 'POD/u_shift.npy'
	else:
		raise

	output_encoder = np.load(args.data_dir + output_encoder_file)
	output_shift = np.load(args.data_dir + output_shift_file)

# Reduce the data and save
reduced_m_data = np.einsum('mr,dm->dr',input_encoder,m_data)
if output_encoder is not None:
	u_data -= output_shift
	reduced_u_data = np.einsum('ur,du->dr',output_encoder,u_data)
else:
	reduced_u_data = u_data

reduced_file_name = mq_file.split('.npz')[0]+'_reduced.npz'

np.savez(args.data_dir+reduced_file_name, m_data=reduced_m_data,q_data=reduced_u_data)

print('Successfully reduced the input-output data.')

J_file = 'JstarPhi_data.npz'
all_J_data = np.load(args.data_dir+J_file)
J_data = all_J_data['JstarPhi_data'].transpose((0,2,1))


# Assuming the J data is already output reduced:
reduced_J_data = np.einsum('dum,mr->dur',J_data,input_decoder)
reduced_file_name = J_file.split('.npz')[0]+'_reduced.npz'
np.savez(args.data_dir+reduced_file_name, J_data=reduced_J_data)

print('Successfully reduced the Jacobian data.')

