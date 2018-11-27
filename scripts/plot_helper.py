import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

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
    
    figure, axis = pyplot.subplots(figsize=(4,4))
    axis.quiver(tails[:,0], tails[:,1], vectors[:,0], vectors[:,1], angles='xy', scale_units='xy', scale=1)
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
    
    figure, (axis1, axis2) = pyplot.subplots(1, 2, figsize=(6,3))
    
    # draw grid lines
    xcolor, ycolor = '#CAB18C', '#005481'
    for i in range(x.size):
        axis1.plot(X[i,:], Y[i,:], c=xcolor, linewidth=1)
        axis2.plot(X_new[i,:], Y_new[i,:], color=xcolor, linewidth=1)
        axis1.plot(X[:,i], Y[:,i], c=ycolor, linewidth=1)
        axis2.plot(X_new[:,i], Y_new[:,i], color=ycolor, linewidth=1)
    
    # draw basis vectors
    origin = numpy.zeros(2)
    identity = numpy.identity(2)
    color = (xcolor, ycolor)
    axis1.quiver(origin, origin, identity[0,:], identity[1,:], color=color, angles='xy', scale_units='xy', scale=1)
    axis2.quiver(origin, origin, matrix[0,:], matrix[1,:], color=color, angles='xy', scale_units='xy', scale=1)
    
    # draw optional vectors
    if vectors:
        red = '#FF3333'
        origin = numpy.zeros(1)
        for vector in vectors:
            axis1.quiver(origin, origin, [vector[0]], [vector[1]], color=red, angles='xy', scale_units='xy', scale=1)
            vector_new = matrix@vector.reshape(-1,1)
            axis2.quiver(origin, origin, [vector_new[0]], [vector_new[1]], color=red, angles='xy', scale_units='xy', scale=1)

    # show x-y axis in the center, hide frames, set xlimit & ylimit
    limit = 4
    for axis in (axis1, axis2):     
        axis.spines['left'].set_position('center')
        axis.spines['bottom'].set_position('center')
        axis.spines['right'].set_color('none')
        axis.spines['top'].set_color('none')
        axis.set_xlim([-limit, limit])
        axis.set_ylim([-limit, limit])
    axis1.set_title('before transformation')
    axis2.set_title('after transformation')

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
    
    figure, (axis1, axis2, axis3) = pyplot.subplots(1, 3, figsize=(9,3))
    
    # draw grid lines
    xcolor, ycolor = '#CAB18C', '#005481'
    for i in range(x.size):
        axis1.plot(X[i,:], Y[i,:], c=xcolor, linewidth=1)
        axis2.plot(X_new1[i,:], Y_new1[i,:], color=xcolor, linewidth=1)
        axis3.plot(X_new2[i,:], Y_new2[i,:], color=xcolor, linewidth=1)
        axis1.plot(X[:,i], Y[:,i], c=ycolor, linewidth=1)
        axis2.plot(X_new1[:,i], Y_new1[:,i], color=ycolor, linewidth=1)
        axis3.plot(X_new2[:,i], Y_new2[:,i], color=ycolor, linewidth=1)
    
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
    axis1.set_title('before transformation')
    axis2.set_title('after 1 transformation')
    axis3.set_title('after 2 transformations')

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

if __name__ == "__main__":
    pass
