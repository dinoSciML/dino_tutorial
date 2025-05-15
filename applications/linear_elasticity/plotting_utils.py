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

import matplotlib.pyplot as plt

import dolfin as dl

def ip_solution_plot(ip_sols, titles=None,out_name = 'ip_comp_plot.pdf'):
    
    fig, axes = plt.subplots(1, len(ip_sols), figsize=(5*len(ip_sols), 5))

    assert len(titles) == len(ip_sols)

    vmin = 1e30
    vmax = -1e30
    for f in ip_sols:
        if isinstance(f, dl.Function):
            fmin = f.vector().min()
            fmax = f.vector().max()
            if fmin < vmin:
                vmin = fmin
            if fmax > vmax:
                vmax = fmax
    images = []
    
    for ip_sol, ax,  title in zip(ip_sols,axes,titles):
        plt.sca(ax)
        image = dl.plot(ip_sol,vmin = vmin, vmax = vmax)
        ax.axis('off')
        ax.set_title(title, fontsize=18)
        images.append(image)
        
        # axes[i].set_title(f"Function {i+1}")
    cbar = fig.colorbar(images[-1], ax=axes, orientation='vertical', fraction=0.05, pad=0.01, shrink=0.5)
    # cbar.set_label("Value")
    # plt.colorbar(cbar)
    
    # plt.tight_layout()
    # plt.draw()
    # plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.show()