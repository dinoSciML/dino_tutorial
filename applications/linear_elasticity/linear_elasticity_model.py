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

import dolfin as dl
import ufl
import math
import numpy as np

import hippylib as hp



def linear_elasticity_settings():
    # Define some basic settings for 
    settings = {'ndim': 2,
                'nx': 40,
                'ny': 20,
                'L' : 2,
                'H' : 1,
                'nx_targets': 10,
                'ny_targets': 5,
                'pointwise_std' : 4.0,
                'correlation_length': 0.2,
                'mean_function_value': 0.3,
                'noise_variance':1e-3}

    return settings

def LinearElasticityPrior(Vh_PARAMETER, pointwise_std, correlation_length, mean=None, anis_diff=None):
    # Delta and gamma
    delta = 1.0 / (pointwise_std * correlation_length)
    gamma = delta * correlation_length ** 2
    if anis_diff is None:
        theta0 = 1
        theta1 = 1
        alpha = math.pi / 4.
        anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=1)
        anis_diff.set(theta0, theta1, alpha)
    if mean is None:
        return hp.BiLaplacianPrior(Vh_PARAMETER, gamma, delta, anis_diff, robin_bc=True)
    else:
        return hp.BiLaplacianPrior(Vh_PARAMETER, gamma, delta, anis_diff, mean=mean, robin_bc=True)

class linear_elasticity_varf:
	def __init__(self,Vh,my_ds,traction,body_force=None,mean = None):
		self.Vh = Vh
		self.my_ds = my_ds
		self.traction = traction
		if body_force is None:
			self.body_force = dl.Constant((0.0,0.0))
		else:
			self.body_force = body_force

		if mean is None:
			self._mean_function = dl.Constant(0.0)
		else:
			self.mean_function = dl.Constant(mean)

	def __call__(self,u,m,p):
		
		# Lame parameters
		E = dl.exp(m) + dl.Constant(1.0)
		nu = 0.4
		mu = E/(2.0*(1.0 + nu))
		lmbda = (E*nu)/((1.0+nu)*(1.0 - 2.0*nu))

		d = u.geometric_dimension()

		eps = dl.sym(dl.grad(u))
		sigma = lmbda * lmbda*dl.tr(eps)*dl.Identity(d) + 2.0*mu*eps
		# Total potential energy:
		Pi = dl.inner(sigma, eps)*dl.dx + dl.dot(self.body_force,u)*dl.dx + dl.dot(self.traction,u)*self.my_ds(1)

		return dl.derivative(Pi,u,p)
	
def linear_elasticity_model(settings):
    # Set up the mesh, finite element spaces, and PDE forward problem
    ndim = settings['ndim']
    nx = settings['nx']
    ny = settings['ny']
    # define geometry params
    L = settings['L']
    H = settings['H']
    
    mesh = dl.RectangleMesh(dl.Point(0., 0.), dl.Point(L, H), nx, ny, "crossed")
    # mesh = dl.UnitSquareMesh(nx, ny)
    Vh2 = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]
    print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(
        Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()) )

    # Now for the PDE formulation

    # Dirichlet boundary conditions

    def left_boundary(x, on_boundary):
        return on_boundary and ( x[0] < dl.DOLFIN_EPS)

    def right_boundary(x, on_boundary):
        return on_boundary and (x[0] > 1.0 - dl.DOLFIN_EPS)

    u_left = dl.Constant((0.0,0.0))

    # u_bdr = dl.Expression("x[1]", degree=1)
    u_bdr0 = dl.Constant((0.0,0.0))
    bc = dl.DirichletBC(Vh[hp.STATE], u_left, left_boundary)
    bc0 = dl.DirichletBC(Vh[hp.STATE], u_bdr0, left_boundary)

    # Traction boundary conditions
    boundary_subdomains = dl.MeshFunction("size_t",mesh,mesh.topology().dim()-1)
    boundary_subdomains.set_all(0)

    dl.AutoSubDomain(right_boundary).mark(boundary_subdomains,1)

    my_ds = dl.ds(subdomain_data = boundary_subdomains)

    right_traction_expr = dl.Expression(("a*exp(-1.0*pow(x[1] - 0.5,2)/b)", "c*(1.0 + (x[1]/d))"),\
                                                 a=0.06, b=4, c=0.03, d=10, degree=5)
    right_t = dl.interpolate(right_traction_expr,Vh[hp.STATE])

    pde_varf = linear_elasticity_varf(Vh,my_ds,right_t)

    # pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    # pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=False)
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0)

    # Set up the prior

    pointwise_std = settings['pointwise_std']
    correlation = settings['correlation_length']
    mean_function_value = settings['mean_function_value']
    mean_function = dl.project(dl.Constant(mean_function_value), 
                            Vh[hp.PARAMETER])

    prior = LinearElasticityPrior(Vh[hp.PARAMETER], pointwise_std,correlation, 
                               mean = mean_function.vector())
 
    # Set up the observation and misfits
    nx_targets = settings['nx_targets']
    ny_targets = settings['ny_targets']
    ntargets = nx_targets*ny_targets

    #Targets only on the bottom
    x_targets = np.linspace(0.1,0.9,nx_targets)*L
    y_targets = np.linspace(0.1,0.9,ny_targets)*H
    targets = []
    for xi in x_targets:
        for yi in y_targets:
            targets.append((xi,yi))
    targets = np.array(targets)
    # print('targets = ',targets)

    print( "Number of observation points: {0}".format(ntargets) )
    B = hp.assemblePointwiseObservation(Vh[hp.STATE], targets)
    misfit = hp.DiscreteStateObservation(B, noise_variance=settings['noise_variance'])
    
    # misfit = hp.PointwiseStateObservation(Vh[hp.STATE], targets)
    model = hp.Model(pde, prior, misfit)
    model.targets = targets
    
    return model

# plot observable:
def get_unorm(model, u):
    U = dl.Function(model.problem.Vh[hp.STATE], u)
    V_x = model.problem.Vh[hp.STATE].sub(0).collapse()
    u_x = dl.interpolate(U.sub(0), V_x)
    u_y = dl.interpolate(U.sub(1), V_x)
    u_norm = dl.Function(V_x)
    u_norm.vector()[:] = np.sqrt(u_x.vector()[:]**2 + u_y.vector()[:]**2)
    return u_norm





class SquareAndQuarterRingPermitivityField(dl.UserExpression):
    """Example log permitivity field composed of a square and ring"""
    def inside_ring(self, x, length):
        dist_sq = (x[0]-0.8*length)**2 + (x[1]-0.2*length)**2
        return (dist_sq <= (0.6*length)**2) and (dist_sq  >= (0.5*length)**2) \
            and x[1] >= 0.2*length and x[0] <= 0.8*length
    
    def inside_square(self, x, length):
        return (x[0]>=0.6*length and x[0] <= 0.8*length \
                and x[1] >= 0.2*length and x[1] <= 0.4*length)
    
    def eval(self, value, x):
        if  self.inside_square(x, 1.0):
            value[0] = 4.0
        else:
            value[0] = -4.0

    def value_shape(self):
        return ()

def true_parameter(prior: hp.prior._Prior, 
                   expr: dl.UserExpression | None = SquareAndQuarterRingPermitivityField()) \
                   -> dl.Vector:
    """Generates the true parameter either as random draw from supplied prior or
    as the supplied user expression 

    :param prior: Prior
    :type prior: hippylib.prior._Prior

    :param expr: Expression for true parameter, defaults to 
        :code:`SquareAndQuarterRingPermitivityField()`
    :type expr: dolfin.UserExpression, optional

    :return: True parameter
    :rtype: dolfin.Vector
    """
    if expr is not None:
        mtrue_func = dl.interpolate(expr, prior.Vh)
        return mtrue_func.vector()
    
    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    hp.parRandom.normal(1.0, noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise, mtrue)
    return mtrue  