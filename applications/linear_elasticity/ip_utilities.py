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

import torch
import torch.optim as opt
from dinotorch_lite.misfits import WeightedQuadraticMisfit


def map_estimate(neural_op, m_r, model, iterations = 100, verbose = False):

    m_r_surr = torch.tensor(m_r.copy()[:], dtype=torch.float32)
    m_r_surr.requires_grad = True
    
    m_r_surr0 = torch.tensor(m_r.copy()[:], dtype=torch.float32)
    m_r_surr0.requires_grad = False
    
    # Optimize
    likelihood = WeightedQuadraticMisfit.from_hippylib(model.misfit)

    regularization = lambda m_r_surr:  torch.linalg.vector_norm(m_r_surr - m_r_surr0, 2)**2 
    lbfgs = opt.LBFGS([m_r_surr], line_search_fn='strong_wolfe')
    
    def closure():
        lbfgs.zero_grad()
        objective = likelihood(q=neural_op(m_r_surr)) + regularization(m_r_surr)
        objective.backward(retain_graph=True)
        return objective
    
    iteration, gradnorm = 0, 1
    lbfgs_history = []
    for iteration in range(iterations):
        lbfgs_history.append({
            'L': likelihood(q=neural_op(m_r_surr)).item(), 
            'R': regularization(m_r_surr)})
        lbfgs.step(closure)
        if iteration %10 == 0 and verbose:
            print(' | '.join([f"Iteration: {iteration}"] + [f"{k}: {v:.3}" for k,v in lbfgs_history[-1].items()]))
    
    q_surr = neural_op(m_r_surr)

    return m_r_surr.detach().numpy()
