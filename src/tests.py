from FEMSolver import FEM
from NodeDists import *
from Meshing import Mesh
import numpy as np
import matplotlib.pyplot as plt
from SampleShapes import *
from PlottingUtils import *


def converge(shape, ns, mpoints):

    disp_errs = np.zeros((mpoints, mpoints, 2, len(ns)))
    force_errs = np.zeros_like(disp_errs)
    
    stress_errs = np.zeros((mpoints, mpoints, 3, len(ns)))
    strain_errs = np.zeros_like(stress_errs)

    nnodes = np.zeros((len(ns)))

    for i, n in enumerate(ns):
        print(f'Solving n={n} shape')
        solver = shape(n)
        solver.solve()
        #solver.show_deformation(magnification=10000)
        solver.get_props(mpoints=mpoints)

        stress_errs[:, :, :, i] = solver.Stresses
        strain_errs[:, :, :, i] = solver.Strains
        disp_errs[:, :, :, i] = solver.Displacements
        force_errs[:, :, :, i] = solver.Forces

        nnodes[i] = solver.mesh.XY.shape[0]


    stress_errs -= stress_errs[:, :, :, -1][:, :, :, None]
    strain_errs -= strain_errs[:, :, :, -1][:, :, :, None]
    disp_errs -= disp_errs[:, :, :, -1][:, :, :, None]
    force_errs -= force_errs[:, :, :, -1][:, :, :, None]
    #errs /= errs[:, :, :, -1][:, :, :, None] + 1.0E-40

    #x, y = solver.coords

    #labels = ["$\epsilon_{11}$ error", "$\epsilon_{22}$ error", "$\gamma_{12}$ error", "$|\epsilon|$ error"]

    #corners = solver.mesh.all_corners

    #corners = np.append(corners, corners[0, :][None, :], axis=0)

    #plotting_3(x, y, errs[:, :, :, 0], labels, corners)

    abs_stress_err = np.linalg.norm(stress_errs, axis=-2)
    abs_strain_err = np.linalg.norm(strain_errs, axis=-2)
    abs_disp_err = np.linalg.norm(disp_errs, axis=-2)
    abs_force_err = np.linalg.norm(force_errs, axis=-2)
    
    
    stress_err = np.average(abs_stress_err.reshape(-1, abs_stress_err.shape[-1]), axis=0)
    strain_err = np.average(abs_strain_err.reshape(-1, abs_strain_err.shape[-1]), axis=0)
    disp_err = np.average(abs_disp_err.reshape(-1, abs_disp_err.shape[-1]), axis=0)
    force_err = np.average(abs_force_err.reshape(-1, abs_force_err.shape[-1]), axis=0)

    plt.plot(nnodes[:-1], stress_err[:-1])
    plt.yscale("log")
    plt.xlabel("Number of Nodes")
    plt.ylabel(f"Log error on normed Stresses")
    plt.title(f"Error convergence of Stress with increasing mesh complexity")
    plt.show()

    plt.plot(nnodes[:-1], strain_err[:-1])
    plt.yscale("log")
    plt.xlabel("Number of Nodes")
    plt.ylabel(f"Log error on normed Strains")
    plt.title(f"Error convergence of Strain with increasing mesh complexity")
    plt.show()

    plt.plot(nnodes[:-1], disp_err[:-1])
    plt.yscale("log")
    plt.xlabel("Number of Nodes")
    plt.ylabel(f"Log error on normed Displacements")
    plt.title(f"Error convergence of Displacement with increasing mesh complexity")
    plt.show()

    plt.plot(nnodes[:-1], force_err[:-1])
    plt.yscale("log")
    plt.xlabel("Number of Nodes")
    plt.ylabel(f"Log error on normed Forces")
    plt.title(f"Error convergence of Force with increasing mesh complexity")
    plt.show()


mpoints = 4000

ns = np.unique(np.logspace(np.log10(2), np.log10(150), 15).astype(int))
print(len(ns))
#solver = simple_shape(100)
#print(solver.mesh.XY.shape[0])
#solver.mesh.plot()
#solver.solve()
#solver.get_props()
#solver.plot_props()

converge(simple_shape, ns, mpoints)