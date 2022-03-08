from Meshing import Mesh
from FEMSolver import FEM
from NodeDists import Uniform
import numpy as np


def simple_shape(n):
    '''
    Simple Quadrilateral test case
    '''
    corners = np.array([[0, 1], # top left
                    [2, 1], # top right
                    [2, 0.5], # bottom right
                    [0, 0]]) # bottom left

    mesh = Mesh(corners, n, n, coord_func=Uniform)

    force = np.array([0, 20])
    edge='top'

    solver = FEM(mesh, 3E7, 0.3, quad_points=5)
    mesh.apply_load(force, edge)
    mesh.pin_edge('left', 0)
    mesh.pin_edge('left', 1)

    return solver


def C_shape(n):
    '''
    'C' shaped solid
    '''


def I_shape(n):
    '''
    'I' shaped solid
    '''