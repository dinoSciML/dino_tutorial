# Linear Elasticity Neural Operator Tutorial

## Installation

E.g., using `conda`

```
conda create -n torchfem -c conda-forge fenics==2019.1.0 matplotlib scipy jupyter 
pip install torch
pip install tqdm
```

Use `pip` to install additional missing packages as needed


### `hippylib`

This tutorial uses the [`hippylib`](https://github.com/hippylib/hippylib/tree/master) package

Either `git clone` and set `HIPPYLIB_PATH` or use pip. I recommend `git`

### `hippyflow`

This tutorial uses the [`hippyflow`](https://github.com/hippylib/hippyflow/tree/master) package for data generation and dimension reduction

Use `git clone` and set `HIPPYFLOW_PATH`.

##

1. To see the setup for the deterministic inverse problem see `LinearElasticityMAP.ipynb`

MAP stands for maximum a posteriori point; i.e., drawing the connection between the most probably point of the Bayesian posterior and the deterministic inversion

2. To train neural operators use the following steps:

```
python generate_data.py # This generates the samples
python compute_coders.py # This computes the reduced bases
python reduce_data.py # This encodes the training data onto the reduced bases
python training_loop.py # This trains a bunch of neural operators

```

For more information on the training see the notebook `LinearElasticityTraining.ipynb`

3. To visualize the results of the training for inverse problems, see the `LinearElasticityIPComparisons.ipynb` notebook. 