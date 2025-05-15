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


def l2_training(model,loss_func,train_loader, validation_loader,\
                     optimizer,lr_scheduler=None,n_epochs = 100, verbose = False):
    device = next(model.parameters()).device

    train_history = {}
    train_history['train_loss_l2'] = []
    train_history['validation_loss_l2'] = []

    for epoch in range(n_epochs):
        # Training
        train_loss = 0
        model.train()
        for batch in train_loader:
            m, u = batch
            m = m.to(device)
            u = u.to(device)
            u_pred = model(m)
            loss = loss_func(u_pred, u)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item() * m.shape[0]

        train_loss /= len(train_loader.dataset)
        
        train_history['train_loss_l2'].append(train_loss)

        # Evaluation
        with torch.no_grad():
            model.eval()
            validation_loss = 0
            for batch in validation_loader:
                m, u = batch
                m = m.to(device)
                u = u.to(device)
                u_pred = model(m)
                loss = loss_func(u_pred, u)
                validation_loss += loss.item() * m.shape[0]
        validation_loss /= len(validation_loader.dataset)

        # Update learning rate if lr_scheduler is provided
        if lr_scheduler is not None:
            lr_scheduler.step(validation_loss)
        if epoch %20 == 0 and verbose:
            print('epoch = ',epoch)
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6e}")
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {validation_loss:.6e}")

    return model, train_history

def h1_training(model,loss_func_l2,loss_func_jac,train_loader, validation_loader,\
                     optimizer,lr_scheduler=None,n_epochs = 100, verbose = False,\
                     mode="forward", jac_weight = 1.0):
    device = next(model.parameters()).device

    def forward_pass(m):
        return model(torch.reshape(m, (-1, m.shape[-1])))

    if mode == "forward":
        jac_func = torch.func.vmap(torch.func.jacfwd(forward_pass))
    elif mode == "reverse":
        jac_func = torch.func.vmap(torch.func.jacrev(forward_pass))
    else:
        raise ValueError("Jacobian mode must be either 'forward' or 'reverse'")

    train_history = {}
    train_history['train_loss_l2'] = []
    train_history['validation_loss'] = []
    train_history['validation_loss_l2'] = []
    train_history['validation_loss_jac'] = []

    for epoch in range(n_epochs):
        # Training
        train_loss = 0
        model.train()
        for batch in train_loader:
            m, u, J = batch
            m = m.to(device)
            u = u.to(device)
            J = J.to(device)
            u_pred = model(m)
            J_pred = jac_func(m)
            loss_l2 = loss_func_l2(u_pred, u)
            loss_jac = loss_func_jac(J_pred, J)
            loss = loss_l2 + jac_weight * loss_jac
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item() * m.shape[0]

        train_loss /= len(train_loader.dataset)
        
        train_history['train_loss_l2'].append(train_loss)

        # Evaluation
        with torch.no_grad():
            model.eval()
            validation_loss = 0
            validation_loss_l2 = 0
            validation_loss_jac = 0
            for batch in validation_loader:
                m, u, J = batch
                m = m.to(device)
                u = u.to(device)
                J = J.to(device)
                u_pred = model(m)
                loss_l2 = loss_func_l2(u_pred, u)
                loss_jac = loss_func_jac(jac_func(m), J)
                loss = loss_l2 + jac_weight * loss_jac
                validation_loss += loss.item() * m.shape[0]
                validation_loss_l2 += loss_l2.item() * m.shape[0]
                validation_loss_jac += loss_jac.item() * m.shape[0]

        validation_loss /= len(validation_loader.dataset) 
        validation_loss_l2 /=len(validation_loader.dataset) 
        validation_loss_jac /= len(validation_loader.dataset) 

        # Update learning rate if lr_scheduler is provided
        if lr_scheduler is not None:
            lr_scheduler.step(validation_loss)

        train_history['validation_loss'].append(validation_loss)
        train_history['validation_loss_l2'].append(validation_loss_l2)
        train_history['validation_loss_jac'].append(validation_loss_jac)

        # # Evaluation
        # with torch.no_grad():
        #     model.eval()
        #     validation_loss = 0
        #     for batch in validation_loader:
        #         m, u, J = batch
        #         m = m.to(device)
        #         u = u.to(device)
        #         u_pred = model(m)
        #         loss = loss_func(u_pred, u)
        #         validation_loss += loss.item() * m.shape[0]
        # validation_loss /= len(validation_loader.dataset)

        # # Update learning rate if lr_scheduler is provided
        # if lr_scheduler is not None:
        #     lr_scheduler.step(validation_loss)
        if epoch %10 == 0 and verbose:
            print('epoch = ',epoch)
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6e}")
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {validation_loss:.6e}")
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss L2: {validation_loss_l2:.6e}")
            print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss Jac: {validation_loss_jac:.6e}")

    return model, train_history
