# "Land on Vector Spaces"
## Practical Linear Algebra with Python

This learning module re-imagines the teaching of linear algebra with a visually rich, intuitive approach, enabled by computing with Python.
We disregard all the rules-based, memorization-heavy instruction of typical (undergraduate) courses in linear algebra.
Instead, we rely on visuals to elucidate the core concepts, and on computational thinking for applying those concepts to useful settings.

#### Lesson 1: Transform all the vectors
What is a vector? The physicist's view versus the computer scientist's view. Fundamental vector operations: visualizing vector addition and multiplication by a scalar. Intuitive presentation of basis vectors, linear combination and span. What is a matrix? A matrix as a linear transformation mapping a vector in one space, to another space. Visualizing linear transformations. Matrix-vector multiplication: a linear combination of the matrix columns. Some special transformations: rotation, shear, scaling. Matrix-matrix multiplication: a composition of two linear transformations. Idea of inverse of a matrix as a transformation that takes vectors back to where they came from.

#### Lesson 2: The matrix is everywhere
A matrix is a linear transformation… visualize it. Norm of a vector. 
A matrix maps a circle to an ellipse… visualize it. A vector that doesn't change direction after a linear transformation is an eigenvalue of the matrix. 
A matrix is a system of equations… visualize it (row perspective) 
Inconsistent and underdetermined systems. 
A matrix is a change of basis… visualize it. An inverse of that matrix will change the vector's coordinates back to original basis. 
Matrices in three-dimensional space: linear transformations in 3D; 3D systems of linear equations; dimension and rank.
Visualize the transformations of rank-deficient matrices.

#### Lesson 3:

Developing on the idea that a matrix can be treated as a linear transformation or a change of basis, notebook 4 visually explains the concept of eigenvalues and eigenvectors: eigenvectors of a matrix only change their scales but not directions after applying the linear transformation, eigenvalues are the corresponding scaling factors. Application: PageRank algorithm.

#### Lesson 4:
Geometrical interpretation of singular value decomposition (SVD). While eigendecomposition is a combination of change of basis and stretching, SVD is a combination of rotation and stretching, which can be treated as a generalization of eigendecomposition.
Example: SVD in image compression. A 2D image can be represented as an array where each pixel is an element of the array. By applying SVD and dropping smaller singular values, we can reconstruct the original image at a lower computational and memory cost.