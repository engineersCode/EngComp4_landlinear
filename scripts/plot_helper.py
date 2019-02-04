import numpy
from numpy.linalg import inv
from math import ceil
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle

pyplot.rc('font', family='serif', size='12')

grey = '#808080'
gold = '#cab18c'
blue = '#005481'
red = '#ff3333'
newblue = '#004065'
lightblue = '#0096d6'
origin = numpy.zeros(1)

def plot_vector(vectors, tails=None):
    ''' draw 2d vectors based on the values of vectors and the position of theirs tails
    vectors: list of 2-element tuples representing 2d vectors
    tails: tail's coordinates of each vector
    '''   
    vectors = numpy.array(vectors)
    assert vectors.shape[1] == 2, "Each vector should have 2 elements."  
    if tails is not None:
        tails = numpy.array(tails)
        assert tails.shape[1] == 2, "Each tail should have 2 elements."
    else:
        tails = numpy.zeros_like(vectors)
    
    # tile vectors or tail array if needed
    nvectors = vectors.shape[0]
    ntails = tails.shape[0]
    if nvectors == 1 and ntails > 1:
        vectors = numpy.tile(vectors, (ntails, 1))
    elif ntails == 1 and nvectors > 1:
        tails = numpy.tile(tails, (nvectors, 1))
    else:
        assert tails.shape == vectors.shape, "vectors and tail must have a same shape"

    # calculate xlimit & ylimit
    heads = tails + vectors
    limit = numpy.max(numpy.abs(numpy.hstack((tails, heads))))
    limit = numpy.ceil(limit * 1.2)   # add some margins
    
    figure, axis = pyplot.subplots(figsize=(6,6))
    axis.quiver(tails[:,0], tails[:,1], vectors[:,0], vectors[:,1], color=newblue, 
                  angles='xy', scale_units='xy', scale=1)
    axis.set_xlim([-limit, limit])
    axis.set_ylim([-limit, limit])
    axis.set_aspect('equal')
    pyplot.grid(True)
    
    # show x-y axis in the center, hide frames
    axis.spines['left'].set_position('center')
    axis.spines['bottom'].set_position('center')
    axis.spines['right'].set_color('none')
    axis.spines['top'].set_color('none')

def plot_linear_transformation(matrix, *vectors):
    """ create line plot and quiver plot to visualize the linear transformation represented by the input matrix
    matrix: (2,2) ndarray
    vectors: optional, other input vectors to visualize
    """
    assert matrix.shape == (2,2), "the input matrix must have a shape of (2,2)"
    
    grid_range = 20
    x = numpy.arange(-grid_range, grid_range+1)
    X, Y = numpy.meshgrid(x,x) 
    X_new = matrix[0,0]*X + matrix[0,1]*Y
    Y_new = matrix[1,0]*X + matrix[1,1]*Y
    
    figure, (axis1, axis2) = pyplot.subplots(1, 2, figsize=(10,5))

    # draw grid lines
    xcolor, ycolor = '#CAB18C', lightblue
    for i in range(x.size):
        axis1.plot(X[i,:], Y[i,:], c=xcolor, linewidth=1)
        axis2.plot(X_new[i,:], Y_new[i,:], color=xcolor, linewidth=1)
        axis1.plot(X[:,i], Y[:,i], c=ycolor, linewidth=1)
        axis2.plot(X_new[:,i], Y_new[:,i], color=ycolor, linewidth=1)
    
    # draw basis vectors
    origin = numpy.zeros(2)
    identity = numpy.identity(2)
    color = (xcolor, ycolor)
    axis1.quiver(origin, origin, identity[0,:], identity[1,:], color=newblue, angles='xy', scale_units='xy', scale=1)
    axis2.quiver(origin, origin, matrix[0,:], matrix[1,:], color=newblue, angles='xy', scale_units='xy', scale=1)
    
    # draw optional vectors
    red, green, orange, purple, brown = '#ff3333', '#559242', '#ffa500', '#a35cff', '#731d1d'
    color_cycle = cycle([red, green, orange, purple, brown])
    if vectors:
        origin = numpy.zeros(1)
        for vector in vectors:
            color = next(color_cycle)
            axis1.quiver(origin, origin, [vector[0]], [vector[1]], color=color, angles='xy', scale_units='xy', scale=1)
            vector_new = matrix@vector.reshape(-1,1)
            axis2.quiver(origin, origin, [vector_new[0]], [vector_new[1]], color=color, angles='xy', scale_units='xy', scale=1)

    # show x-y axis in the center, hide frames, set xlimit & ylimit
    limit = 4
    for axis in (axis1, axis2):     
        axis.spines['left'].set_position('center')
        axis.spines['bottom'].set_position('center')
        axis.spines['right'].set_color('none')
        axis.spines['top'].set_color('none')
        axis.set_xlim([-limit, limit])
        axis.set_ylim([-limit, limit])
    axis1.set_title('Before transformation')
    axis2.set_title('After transformation')

def plot_linear_transformations(matrix1, matrix2):
    """ create line plot and quiver plot to visualize the linear transformations represented by the input matrices
    matrix1: (2,2) ndarray
    matrix2: (2,2) ndarray
    """
    assert matrix1.shape == (2,2), "the first input matrix must have a shape of (2,2)"
    assert matrix2.shape == (2,2), "the second input matrix must have a shape of (2,2)"
    
    grid_range = 20
    x = numpy.arange(-grid_range, grid_range+1)
    X, Y = numpy.meshgrid(x,x) 
    X_new1 = matrix1[0,0]*X + matrix1[0,1]*Y
    Y_new1 = matrix1[1,0]*X + matrix1[1,1]*Y
    X_new2 = matrix2[0,0]*X_new1 + matrix2[0,1]*Y_new1
    Y_new2 = matrix2[1,0]*X_new1 + matrix2[1,1]*Y_new1
    
    figure, (axis1, axis2, axis3) = pyplot.subplots(1, 3, figsize=(12,4))
    
    # draw grid lines
    xcolor, ycolor = gold, blue
    for i in range(x.size):
        axis1.plot(X[i,:], Y[i,:], c=xcolor, linewidth=1)
        axis2.plot(X_new1[i,:], Y_new1[i,:], color=xcolor, linewidth=1, alpha=0.8)
        axis3.plot(X_new2[i,:], Y_new2[i,:], color=xcolor, linewidth=1, alpha=0.8)
        axis1.plot(X[:,i], Y[:,i], c=ycolor, linewidth=1)
        axis2.plot(X_new1[:,i], Y_new1[:,i], color=ycolor, linewidth=1, alpha=0.8)
        axis3.plot(X_new2[:,i], Y_new2[:,i], color=ycolor, linewidth=1, alpha=0.8)
    
    # draw basis vectors
    origin = numpy.zeros(2)
    identity = numpy.identity(2)
    color = (xcolor, ycolor)
    axis1.quiver(origin, origin, identity[0,:], identity[1,:], color=color, angles='xy', scale_units='xy', scale=1)
    axis2.quiver(origin, origin, matrix1[0,:], matrix1[1,:], color=color, angles='xy', scale_units='xy', scale=1)
    axis3.quiver(origin, origin, (matrix2@matrix1)[0,:], (matrix2@matrix1)[1,:], color=color, angles='xy', scale_units='xy', scale=1)
    
    # show x-y axis in the center, hide frames, set xlimit & ylimit
    limit = 4
    for axis in (axis1, axis2, axis3):     
        axis.spines['left'].set_position('center')
        axis.spines['bottom'].set_position('center')
        axis.spines['right'].set_color('none')
        axis.spines['top'].set_color('none')
        axis.set_xlim([-limit, limit])
        axis.set_ylim([-limit, limit])
    axis1.set_title('Before transformation')
    axis2.set_title('After 1 transformation')
    axis3.set_title('After 2 transformations')

def plot_3d_linear_transformation(matrix):
    """ create line plot to visualize the linear transformation represented by the input matrix
    matrix: (3,3) ndarray
    """
    assert matrix.shape == (3,3), "the input matrix must have a shape of (3,3)"

    grid_range = 2
    x = numpy.arange(-grid_range, grid_range+1)
    X, Y, Z = numpy.meshgrid(x,x,x)
    X_new = matrix[0,0]*X + matrix[0,1]*Y + matrix[0,2]*Z
    Y_new = matrix[1,0]*X + matrix[1,1]*Y + matrix[1,2]*Z
    Z_new = matrix[2,0]*X + matrix[2,1]*Y + matrix[2,2]*Z

    figure = pyplot.figure(figsize=(8,4))
    axis1 = figure.add_subplot(1, 2, 1, projection='3d')
    axis2 = figure.add_subplot(1, 2, 2, projection='3d')

    # draw grid lines
    xcolor, ycolor, zcolor = '#0084b6', '#d8a322', '#FF3333'
    linewidth = 1
    for i in range(x.size):
        for j in range(x.size):
            axis1.plot(X[:,i,j], Y[:,i,j], Z[:,i,j], color=xcolor, linewidth=linewidth)
            axis1.plot(X[i,:,j], Y[i,:,j], Z[i,:,j], color=ycolor, linewidth=linewidth)
            axis1.plot(X[i,j,:], Y[i,j,:], Z[i,j,:], color=zcolor, linewidth=linewidth)
            axis2.plot(X_new[:,i,j], Y_new[:,i,j], Z_new[:,i,j], color=xcolor, linewidth=linewidth)
            axis2.plot(X_new[i,:,j], Y_new[i,:,j], Z_new[i,:,j], color=ycolor, linewidth=linewidth)
            axis2.plot(X_new[i,j,:], Y_new[i,j,:], Z_new[i,j,:], color=zcolor, linewidth=linewidth)

    # show x-y axis in the center, hide frames, set xlimit & ylimit
    limit = 2 * 1.20
    axis1.set_xlim([-limit, limit])
    axis1.set_ylim([-limit, limit])
    axis1.set_zlim([-limit, limit])
    axis1.set_title('before transformation')
    axis2.set_title('after transformation')

def plot_basis_helper(axis, I, J, vector=None, title=None):
    """ Plot the new coordinate system determined by the basis I,J.
    axis: 
    I, J: (2, ) numpy array
    vector: vector's coordinates on new basis
    """
    grid_range = 20
    x = numpy.arange(-grid_range, grid_range+1)
    X_, Y_ = numpy.meshgrid(x,x)   # grid coordinates on the new basis
    X = I[0]*X_ + J[0]*Y_   # grid coordinates on the standard basis
    Y = I[1]*X_ + J[1]*Y_
    
    # draw origin
    axis.scatter(origin, origin, c='black', s=6)

    # draw grid lines of the new coordinate system
    lw_grid = 0.4
    for i in range(x.size):
        axis.plot(X[i,:], Y[i,:], c=grey, lw=lw_grid)
        axis.plot(X[:,i], Y[:,i], c=grey, lw=lw_grid)
    
    # highlight new axes (spines)
    lw_spine = 0.5
    zero_id = numpy.where(x==0)[0][0]
    axis.plot(X[zero_id,:], Y[zero_id,:], c=gold, lw=lw_spine)
    axis.plot(X[:,zero_id], Y[:,zero_id], c=blue, lw=lw_spine)

    # draw basis vectors using quiver plot
    axis.quiver(origin, origin, [I[0]], [I[1]], color=gold, angles='xy', scale_units='xy', scale=1)
    axis.quiver(origin, origin, [J[0]], [J[1]], color=blue, angles='xy', scale_units='xy', scale=1)

    # draw input vector on new coordinate system
    bound = 5
    if vector is not None:
        M = numpy.transpose(numpy.vstack((I,J)))
        vector = M @ vector.reshape(-1,1)
        axis.quiver(origin, origin, [vector[0]], [vector[1]], color=red, angles='xy', scale_units='xy', scale=1)
        bound = max(ceil(numpy.max(numpy.abs(vector))), bound)
    
    # hide frames, set xlimit & ylimit, set title
    axis.set_xlim([-bound, bound])
    axis.set_ylim([-bound, bound])
    axis.axis('off')
    if title is not None:
        axis.set_title(title)

    # add text next to new basis vectors
    text_params = {'ha': 'center', 'va': 'center', 'size' : 13}
    axis.text((I[0]-J[0])/2, (I[1]-J[1])/2, '$i$', color=gold, **text_params)
    axis.text((J[0]-I[0])/2, (J[1]-I[1])/2, '$j$', color=blue, **text_params)
    if vector is not None:
        axis.text(vector[0]*1.1, vector[1]*1.1, '$v$', color=red, **text_params)

def plot_basis(I, J, vector):
    """ Plot the vector on the basis defined by I and J
    """
    figure, axis = pyplot.subplots(figsize=(4,4))
    plot_basis_helper(axis, I, J, vector=vector)

def plot_change_basis(I, J, vector):
    """ Create a side-by-side plot of the vector both on the standard basis and on the new basis
    """
    figure, (axis1, axis2) = pyplot.subplots(1, 2, figsize=(8,4))
    M = numpy.transpose(numpy.vstack((I,J)))
    vector_ = inv(M) @ vector.reshape(-1, 1)
    plot_basis_helper(axis1, numpy.array([1,0]), numpy.array([0,1]), vector=vector, title='standard basis')
    plot_basis_helper(axis2, I, J, vector=vector_, title='new basis')

if __name__ == "__main__":
    pass