# "Land on Vector Spaces"
## Practical Linear Algebra with Python

This learning module re-imagines the teaching of linear algebra with a visually rich, intuitive approach, enabled by computing with Python.
We disregard all the rules-based, memorization-heavy instruction of typical (undergraduate) courses in linear algebra.
Instead, we rely on visuals to elucidate the core concepts, and on computational thinking for applying those concepts to useful settings.

#### Lesson 1:
What is a vector? The physicist's view versus the computer scientist's view. Fundamental vector operations: visualizing vector addition and multiplication by a scalar. Intuitive presentation of basis vectors, linear combination and span. What is a matrix? A matrix as a linear transformation mapping a vector in one space, to another space. Visualizing linear transformations. Matrix-vector multiplication: a linear combination of the matrix columns. Some special transformations: rotation, shear, scaling. Matrix-matrix multiplication: a composition of two linear transformations. Idea of inverse of a matrix as a transformation that takes vectors back to where they came from.

#### Lesson 2:
Matrix-vector multiplication as the left-hand side of a linear system of equation. Idea of solving for an unknown vector in a linear system is equivalent to finding the input vector given the transformation matrix its transformed vector (right-hand side vector). Visualize 3D linear transformations. What is rank of a matrix: the dimensionality of the space spanned by the transformed vectors. Visualize the transformations of rank-deficient matrices.

#### Lesson 3:
Matrix-vector multiplication as a change of basis. A matrix converts a vector's coordinates from one coordinate system to another. Visualizing the same vector before and after applying the change of basis. An inverse of that matrix will change the vector's coordinates back to original basis. Differentiate the interpretation of linear transformation with the interpretation of changing basis: the former means a matrix transforms a vector to a new vector under the same basis, the latter means a matrix can express the same vector's coordinates in a new coordinate system (basis).

#### Lesson 4:
Developing on the idea that a matrix can be treated as a linear transformation or a change of basis, notebook 4 visually explains the concept of eigenvalues and eigenvectors: eigenvectors of a matrix only change their scales but not directions after applying the linear transformation, eigenvalues are the corresponding scaling factors. Application: PageRank algorithm.

#### Lesson 5:
Geometrical interpretation of singular value decomposition (SVD). While eigendecomposition is a combination of change of basis and stretching, SVD is a combination of rotation and stretching, which can be treated as a generalization of eigendecomposition.
Example: SVD in image compression. A 2D image can be represented as an array where each pixel is an element of the array. By applying SVD and dropping smaller singular values, we can reconstruct the original image at a lower computational and memory cost.