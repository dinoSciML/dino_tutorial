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
#
# For additional questions contact Thomas O'Leary-Roseberry
#
# Author: Blake Christiersen

from __future__ import annotations


import hippylib as hp
import torch
from torch import nn

class WeightedQuadraticMisfit(nn.Module):
    """Discrete (pointwise) state observation
    
    :param B: Observation operator
    :type B: torch.Tensor
     
    :param d: Observation data
    :type d: torch.Tensor
    
    :param Cn: Noise covariance
    :type Cn: torch.Tensor
    """
    def __init__(self, B: torch.Tensor , d: torch.Tensor, Cn: torch.Tensor):
        """Initializes :code:`WeightedQuadraticMisfit`"""
        super().__init__()
        self.B = B
        self.d = d
        self.Cn = Cn

    @property
    def Cn(self):
        return self._Cn
    
    @Cn.setter
    def Cn(self, value: torch.Tensor):
        self._Cn = value
        match self._Cn.dim():
            case 0 | 1:
                self._solve_Cn = lambda T: T / self._Cn
            case 2:
                # TODO: Create Cholesky solver: https://pytorch.org/docs/stable/generated/torch.cholesky_solve.html
                raise NotImplementedError("Correlated noise covariances not supported")
            case _:
                raise ValueError("Noise covariance must be a scalar, vector, or matrix")

    def forward(self, 
            u: torch.Tensor | None = None, 
            q: torch.Tensor | None = None) \
            -> torch.Tensor:
        """Computes discrete state observation misfit
        
        :param u: State field
        :type u: torch.Tensor, optional
        
        :param q: Discrete state observations
        :type q: torch.Tensor, optional

        :return: Misfit
        :rtype: torch.Tensor
        """
        if u is None and q is None:
            raise ValueError("Either the state or discrete state observations must be supplied")
        
        if q is None:
            q = self.B @ u
        
        e = self.d - q
        return 0.5 * torch.inner(e, self._solve_Cn(e))
    
    @staticmethod
    def from_hippylib(misfit: hp.DiscreteStateObservation,
                      device: torch.device | None = None,
                      dtype: torch.dtype = torch.float32) \
                      -> WeightedQuadraticMisfit:
        """Create :code:`torch` based :code:`WeightedQuadraticMisfit` from 
        :code:`hippylib.WeightedQuadraticMisfit`

        :param misfit: hIPPYlib discrete state observation misfit
        :type misfit: hippylib.WeightedQuadraticMisfit

        :param device: Device, defaults to :code:`None`
        :type device: torch.device, optional

        :param dtype: :code:`torch` datatype, defaults to :code:`torch.float32`
        :param dtype: torch.dtype, optional

        :raises TypeError: If :code:`misfit.noise_variance` is improper type

        :return: :code:`torch` based :code:`WeightedQuadraticMisfit` misfit
        :rtype: WeightedQuadraticMisfit
        """
        B = torch.tensor(misfit.B.array(), device=device, dtype=dtype)
        d = torch.tensor(misfit.d[:], device=device, dtype=dtype)
        match misfit.noise_variance:
            case float() | int():
                Cn = torch.tensor(misfit.noise_variance, device=device, dtype=dtype)
            case dl.Matrix():
                Cn = torch.tensor(misfit.noise_variance.array(), device=device, dtype=dtype)
            case dl.Vector():
                Cn = torch.tensor(misfit.noise_variance[:], device=device, dtype=dtype)
            case _:
                raise TypeError("`misfit.noise_variance` is not a `float`, `int`, `dolfin.Matrix`, or `dolfin.Vector`")
        
        return WeightedQuadraticMisfit(B, d, Cn)




