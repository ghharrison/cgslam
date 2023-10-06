# Derived from python-graphslam, (c) 2020 Jeff Irion and contributors


from collections import defaultdict
from functools import reduce
import multiprocessing
import warnings
import time
import torch
import numba
import numpy as np
import scikits.umfpack
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import skcuda.linalg as sc_linalg
from scipy.sparse import SparseEfficiencyWarning, lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import sksparse.cholmod as cm
import ctypes

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa pylint: disable=unused-import
except ImportError:  # pragma: no cover
    plt = None


warnings.simplefilter("ignore", SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

def parallelReduce(reduceFunc, l, numCPUs, connection=None):
    # print("CPUS:", numCPUs, "\tlen(l):", len(l))
    if numCPUs == 1 or len(l) <= 100:
            returnVal= list(reduceFunc(e) for e in l)
            if connection != None:
                    connection.send(returnVal)
            return returnVal

    parent1, child1 = multiprocessing.Pipe()
    parent2, child2 = multiprocessing.Pipe()
    p1 = multiprocessing.Process(target=parallelReduce, args=(reduceFunc, l[:len(l) // 2], numCPUs // 2, child1, ) )
    p2 = multiprocessing.Process(target=parallelReduce, args=(reduceFunc, l[len(l) // 2:], numCPUs // 2 + numCPUs%2, child2, ) )
    p1.start()
    p2.start()
    leftReturn, rightReturn = parent1.recv(), parent2.recv()
    p1.join()
    p2.join()
    returnVal = leftReturn + rightReturn
    if connection != None:
            connection.send(returnVal)
    return returnVal

# cuSparse
_libcusparse = ctypes.cdll.LoadLibrary('libcusparse.so')
_libcusparse.cusparseCreate.restype = int
_libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]

_libcusparse.cusparseDestroy.restype = int
_libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]

_libcusparse.cusparseCreateMatDescr.restype = int
_libcusparse.cusparseCreateMatDescr.argtypes = [ctypes.c_void_p]


# cuSOLVER
_libcusolver = ctypes.cdll.LoadLibrary('libcusolver.so')

_libcusolver.cusolverSpCreate.restype = int
_libcusolver.cusolverSpCreate.argtypes = [ctypes.c_void_p]

_libcusolver.cusolverSpDestroy.restype = int
_libcusolver.cusolverSpDestroy.argtypes = [ctypes.c_void_p]

_libcusolver.cusolverSpDcsrlsvqr.restype = int
_libcusolver.cusolverSpDcsrlsvqr.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_double,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p]

def cholsolve(A, b):
    factor = cm.cholesky(A)
    x = factor(b)
    return x

def cuspsolve(A, b):
    Acsr = csr_matrix(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.empty_like(b)

    # Copy arrays to GPU
    dcsrVal = gpuarray.to_gpu(Acsr.data)
    dcsrColInd = gpuarray.to_gpu(Acsr.indices)
    dcsrIndPtr = gpuarray.to_gpu(Acsr.indptr)
    dx = gpuarray.to_gpu(x)
    db = gpuarray.to_gpu(b)

    # Create solver parameters
    m = ctypes.c_int(Acsr.shape[0])  # Assumes A is square
    nnz = ctypes.c_int(Acsr.nnz)
    descrA = ctypes.c_void_p()
    reorder = ctypes.c_int(0)
    tol = ctypes.c_double(1e-10)
    singularity = ctypes.c_int(0)  # -1 if A not singular

    # create cusparse handle
    timestart = time.time()
    _cusp_handle = ctypes.c_void_p()
    status = _libcusparse.cusparseCreate(ctypes.byref(_cusp_handle))
    assert(status == 0)
    cusp_handle = _cusp_handle.value
    # print(f"\t\tCreating cusparsetook {time.time() - timestart}s")

    # create MatDescriptor
    timestart = time.time()
    status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(descrA))
    assert(status == 0)
    # print(f"\t\tCreating matrix descriptor took {time.time() - timestart}s")

    #create cusolver handle
    timestart = time.time()
    _cuso_handle = ctypes.c_void_p()
    status = _libcusolver.cusolverSpCreate(ctypes.byref(_cuso_handle))
    assert(status == 0)
    cuso_handle = _cuso_handle.value
    # print(f"\t\tCreating cusolverSpCreate took {time.time() - timestart}s")

    # Solve
    timestart = time.time()
    res=_libcusolver.cusolverSpDcsrlsvqr(cuso_handle,
                                         m,
                                         nnz,
                                         descrA,
                                         int(dcsrVal.gpudata),
                                         int(dcsrIndPtr.gpudata),
                                         int(dcsrColInd.gpudata),
                                         int(db.gpudata),
                                         tol,
                                         reorder,
                                         int(dx.gpudata),
                                         ctypes.byref(singularity))
    assert(res == 0)
    if singularity.value != -1:
        raise ValueError('Singular matrix!')
    x = dx.get()  # Get result as numpy array
    # print(f"\t\tSolving took {time.time() - timestart}s")

    # Destroy handles
    timestart = time.time()
    status = _libcusolver.cusolverSpDestroy(cuso_handle)
    assert(status == 0)
    status = _libcusparse.cusparseDestroy(cusp_handle)
    assert(status == 0)
    # print(f"\t\tDestroying handles {time.time() - timestart}s")

    # Return result
    return x

def skcudasolve(A,b):
    sc_linalg.init()
    A_gpu = gpuarray.to_gpu(A.todense())
    b_gpu = gpuarray.to_gpu(b)
    sc_linalg.cho_solve(A_gpu, b_gpu)
    return b_gpu.get()

def numpysolve(A, b):
    return np.linalg.solve(A, b)

def njitsolve(A, b):
    r"""Solve Ax=b using numpy.linalg.solve with Numba.
    This is done on the CPU, unlike the Torch approach.

    Parameters
    ----------
    A : lil_matrix
        The sparse Hessian matrix. We'll convert this to a (dense) Tensor.
    b : ndarray
        -(gradient). 
    """
    solver = numba.njit(numpysolve, fastmath=True)
    return solver(A.todense(), b)


# pylint: disable=too-few-public-methods
class _Chi2GradientHessian:
    r"""A class that is used to aggregate the :math:`\chi^2` error, gradient, and Hessian.

    Parameters
    ----------
    dim : int
        The compact dimensionality of the poses

    Attributes
    ----------
    chi2 : float
        The :math:`\chi^2` error
    dim : int
        The compact dimensionality of the poses
    gradient : defaultdict
        The contributions to the gradient vector
    hessian : defaultdict
        The contributions to the Hessian matrix

    """
    def __init__(self, dim):
        self.chi2 = 0.
        self.dim = dim
        self.gradient = defaultdict(lambda: np.zeros(dim))
        self.hessian = defaultdict(lambda: np.zeros((dim, dim)))

    @staticmethod
    def update(chi2_grad_hess, incoming):
        r"""Update the :math:`\chi^2` error and the gradient and Hessian dictionaries.

        Parameters
        ----------
        chi2_grad_hess : _Chi2GradientHessian
            The ``_Chi2GradientHessian`` that will be updated
        incoming : tuple
            TODO

        """
        chi2_grad_hess.chi2 += incoming[0]

        for idx, contrib in incoming[1].items():
            chi2_grad_hess.gradient[idx] += contrib

        for (idx1, idx2), contrib in incoming[2].items():
            if idx1 <= idx2:
                chi2_grad_hess.hessian[idx1, idx2] += contrib
            else:
                chi2_grad_hess.hessian[idx2, idx1] += np.transpose(contrib)

        return chi2_grad_hess


class Graph(object):
    r"""A graph that will be optimized via Graph SLAM.

    Parameters
    ----------
    edges : list[graphslam.edge.base_edge.BaseEdge]
        A list of the vertices in the graph
    vertices : list[graphslam.vertex.Vertex]
        A list of the vertices in the graph

    Attributes
    ----------
    _chi2 : float, None
        The current :math:`\chi^2` error, or ``None`` if it has not yet been computed
    _edges : list[graphslam.edge.base_edge.BaseEdge]
        A list of the edges (i.e., constraints) in the graph
    _fixed_vertices : set[int]
        The set of vertices that are fixed
    _gradient : numpy.ndarray, None
        The gradient :math:`\mathbf{b}` of the :math:`\chi^2` error, or ``None`` if it has not yet been computed
    _hessian : scipy.sparse.lil_matrix, None
        The Hessian matrix :math:`H`, or ``None`` if it has not yet been computed
    _vertices : list[graphslam.vertex.Vertex]
        A list of the vertices in the graph

    """
    def __init__(self, edges, vertices):
        # The vertices and edges lists
        self._edges = edges
        self._vertices = vertices
        self._fixed_vertices = set()

        # The chi^2 error, gradient, and Hessian
        self._chi2 = None
        self._gradient = None
        self._hessian = None

        self.total_solve_time = 0.
        self.total_opt_time = 0.

        self._link_edges()

    def _link_edges(self):
        """Fill in the ``vertices`` attributes for the graph's edges.

        """
        index_id_dict = {i: v.id for i, v in enumerate(self._vertices)}
        id_index_dict = {v_id: v_index for v_index, v_id in index_id_dict.items()}

        # Fill in the vertices' `index` attribute
        for v in self._vertices:
            v.index = id_index_dict[v.id]

        for e in self._edges:
            e.vertices = [self._vertices[id_index_dict[v_id]] for v_id in e.vertex_ids]

    def calc_chi2(self):
        r"""Calculate the :math:`\chi^2` error for the ``Graph``.

        Returns
        -------
        float
            The :math:`\chi^2` error

        """
        self._chi2 = sum((e.calc_chi2() for e in self._edges))
        return self._chi2
    

    def _calc_edge_chi2_gradient_hessian(self):
        return parallelReduce((lambda e: e.calc_chi2_gradient_hessian()), self._edges, multiprocessing.cpu_count())

    def _calc_chi2_gradient_hessian(self):
        r"""Calculate the :math:`\chi^2` error, the gradient :math:`\mathbf{b}`, and the Hessian :math:`H`.

        """
        n = len(self._vertices)
        dim = len(self._vertices[0].pose.to_compact())
        # edge_chi2_gradient_hessian = self._calc_edge_chi2_gradient_hessian()
        chi2_gradient_hessian = reduce(_Chi2GradientHessian.update, (e.calc_chi2_gradient_hessian() for e in self._edges), _Chi2GradientHessian(dim))
        # chi2_gradient_hessian = reduce(_Chi2GradientHessian.update, edge_chi2_gradient_hessian, _Chi2GradientHessian(dim))

        self._chi2 = chi2_gradient_hessian.chi2

        # Fill in the gradient vector
        self._gradient = np.zeros(n * dim, dtype=np.float64)
        for idx, contrib in chi2_gradient_hessian.gradient.items():
            # If a vertex is fixed, its block in the gradient vector is zero and so there is nothing to do
            if idx not in self._fixed_vertices:
                self._gradient[idx * dim: (idx + 1) * dim] += contrib

        # Fill in the Hessian matrix
        self._hessian = lil_matrix((n * dim, n * dim), dtype=np.float64)
        for (row_idx, col_idx), contrib in chi2_gradient_hessian.hessian.items():
            if row_idx in self._fixed_vertices or col_idx in self._fixed_vertices:
                # For fixed vertices, the diagonal block is the identity matrix and the off-diagonal blocks are zero
                if row_idx == col_idx:
                    self._hessian[row_idx * dim: (row_idx + 1) * dim, col_idx * dim: (col_idx + 1) * dim] = np.eye(dim)
                continue

            self._hessian[row_idx * dim: (row_idx + 1) * dim, col_idx * dim: (col_idx + 1) * dim] = contrib

            if row_idx != col_idx:
                self._hessian[col_idx * dim: (col_idx + 1) * dim, row_idx * dim: (row_idx + 1) * dim] = np.transpose(contrib)

    def torchsolve(self, A, b):
        r"""Solve Ax=b using torch.linalg.solve with CUDA.
        The intention is to make this significantly faster than doing so
        on the CPU using the existing techniques; however doing so does not
        take advantage of sparsity so performance benefit remains to be seen.

        Parameters
        ----------
        A : lil_matrix
            The sparse Hessian matrix. We'll convert this to a (dense) Tensor.
        b : ndarray
            -(gradient). 
        """
        with torch.no_grad():
            return torch.linalg.solve(torch.tensor(A.todense()).cuda(), torch.tensor(b).cuda()).detach().cpu().numpy()
           
    def umfpacksolve(self, A, b):
        return scikits.umfpack.spsolve(A, b)

    def optimize(self, tol=1e-4, max_iter=20, fix_first_pose=True, solver=None):
        r"""Optimize the :math:`\chi^2` error for the ``Graph``.

        Parameters
        ----------
        tol : float
            If the relative decrease in the :math:`\chi^2` error between iterations is less than ``tol``, we will stop
        max_iter : int
            The maximum number of iterations
        fix_first_pose : bool
            If ``True``, we will fix the first pose
        solver : str
            The linear solver to use to optimize the graph. Choices are None (default), "torch", "njit", "numpy", "umfpack"?
        """
        n = len(self._vertices)
        if solver not in ["torch", "njit", "numpy", "umfpack", "skcuda", "cusparse"]:
            print("Optimizing with default linear solver (spsolve)")
        else:
            print("Optimizing with linear solver", solver)

        if fix_first_pose:
            self._vertices[0].fixed = True

        # Populate the set of fixed vertices
        self._fixed_vertices = {i for i, v in enumerate(self._vertices) if v.fixed}

        # Previous iteration's chi^2 error
        chi2_prev = -1.

        # self.plot(title="Initial")
        # plt.show()

        # For displaying the optimization progress
        print("\nIteration                chi^2        rel. change         time (calc)         time (solve)    time(relink)      time(tot.)")
        print("---------                -----        -----------         -----------         ------------    ------------      ----------")

        for i in range(max_iter):
            iter_start_time = time.time()
            self._calc_chi2_gradient_hessian()
            if i == 0:
                print("\tHessian is", self._hessian.todense().shape)

            # Check for convergence (from the previous iteration); this avoids having to calculate chi^2 twice
            if i > 0:
                rel_diff = (chi2_prev - self._chi2) / (chi2_prev + np.finfo(float).eps)
                # print("{:9d} {:20.4f} {:18.6f}".format(i, self._chi2, -rel_diff))
                progress_str = "{:9d} {:20.4f} {:18.6f}".format(i, self._chi2, -rel_diff)
                if self._chi2 <= chi2_prev and rel_diff < tol:
                    print(progress_str)
                    return
            else:
                # print("{:9d} {:20.4f}".format(i, self._chi2))
                progress_str = "{:9d} {:20.4f} \t\t\t".format(i, self._chi2)
                # exit()

            # Update the previous iteration's chi^2 error
            chi2_prev = self._chi2
            iter_solve_begin = time.time()
            # Solve for the updates
            if solver == "torch":
                dx = self.torchsolve(self._hessian, -self._gradient)
            elif solver == "njit":
                dx = njitsolve(self._hessian, -self._gradient)
            elif solver == "numpy":
                dx = numpysolve(self._hessian.todense(), -self._gradient)
            elif solver == "umfpack":
                dx = self.umfpacksolve(self._hessian, -self._gradient)
            elif solver == "skcuda":
                dx = skcudasolve(self._hessian, -self._gradient)
            elif solver == "cusparse":
                dx = cuspsolve(self._hessian, -self._gradient)
            elif solver == "cholmod":
                dx = cholsolve(self._hessian, -self._gradient)
            else:
                dx = spsolve(self._hessian, -self._gradient)  # pylint: disable=invalid-unary-operand-type
            
            iter_solve_end = time.time()
            # Apply the updates
            for v, dx_i in zip(self._vertices, np.split(dx, n)):
                v.pose += dx_i
            self._link_edges()
            # self.plot(title=f"after Iteration {i}")
            # plt.show()
            iter_end_time = time.time()
            progress_str += "\t\t{:6.4f}".format(iter_solve_begin - iter_start_time)
            progress_str += "\t\t{:6.4f}".format(iter_solve_end - iter_solve_begin)
            progress_str += "\t\t{:6.4f}".format(iter_end_time - iter_solve_end)
            progress_str += "\t\t{:6.4f}".format(iter_end_time - iter_start_time)

            self.total_solve_time += (iter_solve_end - iter_solve_begin)
            self.total_opt_time += (iter_end_time - iter_start_time)
            print(progress_str)
        # If we reached the maximum number of iterations, print out the final iteration's results
        self.calc_chi2()
        rel_diff = (chi2_prev - self._chi2) / (chi2_prev + np.finfo(float).eps)
        print("{:9d} {:20.4f} {:18.6f}".format(max_iter, self._chi2, -rel_diff))

    def to_g2o(self, outfile):
        """Save the graph in .g2o format.

        Parameters
        ----------
        outfile : str
            The path where the graph will be saved

        """
        with open(outfile, 'w') as f:
            for v in self._vertices:
                f.write(v.to_g2o())

            for e in self._edges:
                f.write(e.to_g2o())

    def plot(self, vertex_color='r', vertex_marker='o', vertex_markersize=3, edge_color='b', title=None, savepath=None):
        """Plot the graph.

        Parameters
        ----------
        vertex_color : str
            The color that will be used to plot the vertices
        vertex_marker : str
            The marker that will be used to plot the vertices
        vertex_markersize : int
            The size of the plotted vertices
        edge_color : str
            The color that will be used to plot the edges
        title : str, None
            The title that will be used for the plot

        """
        if plt is None:  # pragma: no cover
            raise NotImplementedError

        fig = plt.figure()
        if len(self._vertices[0].pose.position) == 3:
            fig.add_subplot(111, projection='3d')

        for e in self._edges:
            e.plot(edge_color)

        for v in self._vertices:
            v.plot(vertex_color, vertex_marker, vertex_markersize)

        if title:
            plt.title(title)
        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')
        if savepath is not None:
            plt.savefig(savepath, transparent=False, facecolor='white')
            plt.close()
        else:
            plt.draw()
