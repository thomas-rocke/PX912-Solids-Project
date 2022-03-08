from FEMSolver import FEM
from NodeDists import *
from Meshing import Mesh
import numpy as np
import matplotlib.pyplot as plt
from SampleShapes import *
from PlottingUtils import *


mpoints = 4000

ns = [2, 5, 10]

errs = np.zeros((mpoints, mpoints, 3, len(ns)))

dat = 'stresses'


for i, n in enumerate(ns):
    print(f'Solving n={n} shape')
    solver = simple_shape(n)

    solver.solve()
    #solver.show_deformation(magnification=10000)
    solver.get_props(mpoints=mpoints)

    errs[:, :, :, i] = solver.__getattribute__(dat)


errs -= errs[:, :, :, -1]
errs /= errs[:, :, :, -1]

errs = np.abs(errs)
x, y = solver.coords

labels = ["$\epsilon_{11}$ error", "$\epsilon_{22}$ error", "$\gamma_{12}$ error", "$|\epsilon|$ error"]

corners = solver.mesh.all_corners

corners = np.append(corners, corners[0, :], axis=0)

plotting_3(x, y, errs[:, :, :, 0], labels, corners)
plt.show()


