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
import dolfin as dl
import math
import ufl
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla 
import scipy.linalg 

import matplotlib.pyplot as plt
import math
import time

sys.path.append( os.environ.get('HIPPYLIB_PATH', "...") )
import hippylib as hp

sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
import hippyflow as hf

sys.path.append('../')

from linear_elasticity_model import *

def _weighted_l2_norm_vector(x, W):
    Wx = W @ x 
    norm2 = np.einsum('ij,ij->j', Wx, x)
    return np.sqrt(norm2)

################################################################################
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', '--data_dir', type=str, default='data/pointwise/', help="Where to save")
parser.add_argument('-basis_type', '--basis_type', type=str, default='as', help="pod as or kle")
parser.add_argument('-rank', '--rank', type=int, default=400, help="Active subspace rank")
parser.add_argument('-oversample', '--oversample', type=int, default=10, help="Active subspace oversample")
parser.add_argument('-ndata', '--ndata', type=int, default=800, help="Number of samples")

args = parser.parse_args()

data_dir = args.data_dir

################################################################################
# Parameters

rank = args.rank
oversample = args.oversample

################################################################################
# Set up the model

settings = linear_elasticity_settings()
model = linear_elasticity_model(settings)

Vh = model.problem.Vh

prior = model.prior

assert dl.MPI.comm_world.size == 1, print('Not thought out in other cases yet')


if args.basis_type.lower() == 'kle':
	KLE = hf.KLEProjector(prior)
	KLE.parameters['rank'] = rank
	KLE.parameters['oversampling'] = oversample
	KLE.parameters['save_and_plot'] = False

	d_KLE, kle_decoder, kle_encoder = KLE.construct_input_subspace()

	input_decoder = hf.mv_to_dense(kle_decoder)
	input_encoder = hf.mv_to_dense(kle_encoder)


	check_orth = True
	if check_orth:
		PsistarPsi = input_encoder.T@input_decoder
		orth_error = np.linalg.norm(PsistarPsi - np.eye(PsistarPsi.shape[0]))
		print('||Psi^*Psi - I|| = ',orth_error)
		assert orth_error < 1e-5

	np.save(args.data_dir+'KLE_decoder',input_decoder)
	np.save(args.data_dir+'KLE_d',d_KLE)
	np.save(args.data_dir+'KLE_encoder',input_encoder)

elif args.basis_type.lower() == 'as':
	################################################################################
	# Load the data

	data_dir = args.data_dir
	all_data = np.load(data_dir+'mq_data.npz')
	all_data = np.load(data_dir+'mq_data.npz')
	JTPhi_data = np.load(data_dir+'JstarPhi_data.npz')

	m_data = all_data['m_data'][:args.ndata]
	q_data = all_data['q_data'][:args.ndata]
	PhiTJ_data = np.transpose(JTPhi_data['JstarPhi_data'], (0,2,1))[:args.ndata]

	print('m_data.shape = ',m_data.shape)
	print('q_data.shape = ',q_data.shape)
	print('PhistarJ_data.shape = ',PhiTJ_data.shape)


	################################################################################
	# Instance JTJ operator 
	print('Loading JTJ')
	JTJ_operator = hf.MeanJTJfromDataOperator(PhiTJ_data,prior)
	# Set up the Gaussian random
	m_vector = dl.Vector()
	JTJ_operator.init_vector_lambda(m_vector,0)
	Omega = hp.MultiVector(m_vector,rank+oversample)
	hp.parRandom.normal(1.,Omega)

	t0 = time.time()
	print('Beginning doublePassG')
	if hasattr(prior, "R"):
		d_GN, V_GN = hp.doublePassG(JTJ_operator,\
			prior.R, prior.Rsolver, Omega,rank,s=1)
	else:
		d_GN, V_GN = hp.doublePassG(JTJ_operator,\
			prior.Hlr, prior.Hlr, Omega,rank,s=1)

	print('doublePassG took ',time.time() - t0,'s')

	input_decoder = hf.mv_to_dense(V_GN)

	# Compute the projector RV_r from the basis
	RV_GN = hp.MultiVector(V_GN[0],V_GN.nvec())
	RV_GN.zero()
	hp.MatMvMult(prior.R,V_GN,RV_GN)

	input_encoder = hf.mv_to_dense(RV_GN)

	check_orth = True
	if check_orth:
		PsistarPsi = input_encoder.T@input_decoder
		orth_error = np.linalg.norm(PsistarPsi - np.eye(PsistarPsi.shape[0]))
		print('||Psi^*Psi - I|| = ',orth_error)
		assert orth_error < 1e-5

	np.save(args.data_dir+'AS_input_decoder',input_decoder)
	np.save(args.data_dir+'AS_d_GN',d_GN)
	np.save(args.data_dir+'AS_input_encoder',input_encoder)

	fig, ax = plt.subplots()
	ax.semilogy(np.arange(len(d_GN)), d_GN)

	ax.set(xlabel='index', ylabel='eigenvalue',
		   title='GEVP JstarJ spectrum')
	ax.grid()

	fig.savefig(args.data_dir+"JstarJ_eigenvalues.pdf")

else: 
	raise






