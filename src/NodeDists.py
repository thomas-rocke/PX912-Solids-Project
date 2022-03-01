'''
Set of functions to generate real-space nodal coordinates given bounds and number of nodes

'''

import numpy as np

def Transform(corners, Meshgrid, nx, ny):
    '''
    Transforms the square Meshgrid to fit the quadrilateral defined by corners
    Then converts to a form usable by the Mesh class
    '''
    # Unpack
    x, y = Meshgrid

    # Preserve unmodified grid
    x_prime = x.copy()
    y_prime = y.copy()

    # Find base and height of target shape
    base = corners[2, 0] - corners[3, 0]
    height = corners[0, 1] - corners[3, 1]

    # Match Scale
    x *= base
    y *= height

    # Match Left and Bottom edges
    dx = (corners[0, 0] - corners[3, 0])
    dy = (corners[2, 1] - corners[3, 1])

    for i in range(nx):
        for j in range(ny):
            x[i, j] += dx * y_prime[i, j]
            y[i, j] += dy * x_prime[i, j]

    # Match Top and Right edges
    dx = (corners[1, 0] - x[-1, -1])
    dy = (corners[1, 1] - y[-1, -1])
    square = (x_prime * y_prime)

    for i in range(nx):
        for j in range(ny):
            x[i, j] += dx * square[i, j]
            y[i, j] += dy * square[i, j]
    
    XY = np.zeros((nx * ny, 2))

    for i in range(nx):
        for j in range(ny):
            XY[i + nx * j, :] = [x[i, j], y[i, j]]

    return XY




def Uniform(corners, nx, ny):
    '''
    Generates a uniformly distributed meshgrid for the quadrilateral defined by corners
    '''
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    grid = np.meshgrid(x, y)
    return Transform(corners, grid, nx, ny)


def CornerBias(corners, nx, ny):
    '''
    Generates a meshgrid biased towards the corners of the domain
    '''
    # Generate unit grid
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    l = 0.5

    # Apply sqrt biasing, preserving sign
    x = x * np.sqrt(np.abs(x)/l) / np.abs(x + 1E-20)
    y = y * np.sqrt(np.abs(y)/l) / np.abs(y + 1E-20)

    # Rescale and Recenter to unit grid
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)

    x /= x_range
    y /= y_range

    x -= x[0]
    y -= y[0]

    grid = np.meshgrid(x, y)
    return Transform(corners, grid, nx, ny)


