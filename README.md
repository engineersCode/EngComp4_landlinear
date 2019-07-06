# "Land on Vector Spaces"
## Practical Linear Algebra with Python

This learning module re-imagines the teaching of linear algebra with a visually rich, intuitive approach, enabled by computing with Python.
We disregard all the rules-based, memorization-heavy instruction of typical (undergraduate) courses in linear algebra.
Instead, we rely on visuals to elucidate the core concepts, and on computational thinking for applying those concepts to useful settings.

#### [Lesson 1](https://go.gwu.edu/engcomp4lesson1): Transform all the vectors
What is a vector? The physicist's view versus the computer scientist's view. Fundamental vector operations: visualizing vector addition and multiplication by a scalar. Intuitive presentation of basis vectors, linear combination and span. What is a matrix? A matrix as a linear transformation mapping a vector in one space, to another space. Visualizing linear transformations. Matrix-vector multiplication: a linear combination of the matrix columns. Some special transformations: rotation, shear, scaling. Matrix-matrix multiplication: a composition of two linear transformations. Idea of inverse of a matrix as a transformation that takes vectors back to where they came from.

#### [Lesson 2](https://go.gwu.edu/engcomp4lesson2): The matrix is everywhere
A matrix is a linear transformation… visualize it. Norm of a vector. 
A matrix maps a circle to an ellipse… visualize it. A vector that doesn't change direction after a linear transformation is an eigenvector of the matrix. 
A matrix is a system of equations… visualize it (row perspective). 
Inconsistent and underdetermined systems. 
A matrix is a change of basis… visualize it. An inverse of that matrix will change the vector's coordinates back to the original basis. 
Matrices in three-dimensional space: linear transformations in 3D; 3D systems of linear equations; dimension and rank.
Visualize the transformations of rank-deficient matrices.

#### [Lesson 3](https://go.gwu.edu/engcomp4lesson3): Eigenvectors for the win

Geometry of eigendecomposition. Eigenvectors revisited: a matrix transforms a circle to an ellipse, whose semimajor and semiminor axes align with the eigenvectors. 
Composition of scaling transformation and a rotation transformation: not enough! Complete the composition. 
Symmetric matrices, orthogonal eigenvectors. 
Eigendecomposition in general. Diagonalizable matrices. Similar matrices. Eigendecomposition is similarity via a change of basis. 
Compute eigenthings in Python, using NumPy or SymPy. 
Eigenvalues in ecology: matrix population models. 
Markov chains. 
PageRank algorithm.

#### Lesson 4:
Geometrical interpretation of singular value decomposition (SVD). While eigendecomposition is a combination of change of basis and stretching, SVD is a combination of rotation and stretching, which can be treated as a generalization of eigendecomposition.
Example: SVD in image compression. A 2D image can be represented as an array where each pixel is an element of the array. By applying SVD and dropping smaller singular values, we can reconstruct the original image at a lower computational and memory cost.

## Install the Dependencies
The notebooks are compatible with Python version 3.5 or later, all packages required are listed in `environment.yml` and `requirements.txt`. Pick any option below to install the depencies:

#### Create a `conda` environment
Use `environment.yml` to create a `conda` environment:

```console
conda env create -f environment.yml
```

This creates a `conda` environment named "landlinear". For `conda` 4.6 and later versions, activate this environment with

```console
conda activate landlinear
```

For `conda` version prior to 4.6, run `source activate landlinear` on Linux and MacOS, `activate landlinear` on Windows.

#### Install with `conda`

```console
conda install matplotlib numpy scipy jupyter imageio ipywidgets
```

#### Install with `pip`

Install the packages using `requirements.txt` for `pip`:
```console
pip install -r requirements.txt
```

## Check Installation
Run the script `check_install.py` to check whether the required packages are installed correctly:

```console
python check_install.py
```
