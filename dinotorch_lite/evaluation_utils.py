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
import numpy as np

from .dinotorch_utils import *

def evaluate_l2_error(model,validation_loader):

    device = next(model.parameters()).device

    with torch.no_grad():
        model.eval()
        validation_error = 0
        validation_denom = 0
        for batch in validation_loader:
            m, u = batch
            m = m.to(device)
            u = u.to(device)
            u_pred = model(m)
            error = squared_f_error(u_pred, u)
            denom = squared_f_error(0*u_pred, u)
            validation_error += error.item() * m.shape[0]
            validation_denom += denom.item() * m.shape[0]

        rel_error = np.sqrt(validation_error/validation_denom)

    return rel_error