"""
Example of the Python binding of the Gpufit library which implements
Levenberg Marquardt curve fitting in CUDA
https://github.com/gpufit/Gpufit
http://gpufit.readthedocs.io/en/latest/bindings.html#python

Multiple fits of a 2D Gaussian peak function with Poisson distributed noise
This example additionally requires numpy.
"""

import cunumeric as np
import numpy as nnp
import cupy as cp
import pygpufit.gpufit as gf
import pycpufit.cpufit as cf

from legate.core.task import task, InputStore, OutputStore
from legate.core import get_legate_runtime, LegateDataInterface, LogicalStore, StoreTarget
from typing import Optional, Union

def generate_gauss_2d(p, xi, yi):
    """
    Generates a 2D Gaussian peak.
    http://gpufit.readthedocs.io/en/latest/api.html#gauss-2d

    :param p: Parameters (amplitude, x,y center position, width, offset)
    :param xi: x positions
    :param yi: y positions
    :return: The Gaussian 2D peak.
    """

    arg = -(np.square(xi - p[1]) + np.square(yi - p[2])) / (2 * p[3] * p[3])
    y = p[0] * np.exp(arg) + p[4]

    return y

def get_store(obj: LegateDataInterface) -> LogicalStore:
    iface = obj.__legate_data_interface__
    assert iface["version"] == 1
    data = iface["data"]
    # There should only be one field
    assert len(data) == 1
    field = next(iter(data))
    assert not field.nullable
    column = data[field]
    assert not column.nullable
    return column.data

# @task(
#     variants=("cpu", "gpu")
# )
# def fit_wrapper(data: InputStore,
#                 weights: Union[InputStore, None],
#                 model_id: int,
#                 initial_parameters: InputStore, _: OutputStore,
#                 tolerance: Optional[float] = None,
#                 max_number_iterations: Optional[int] = None,
#                 parameters_to_fit: Optional[InputStore] = None,
#                 estimator_id: Optional[int] = None,
#                 user_info: Optional[InputStore] = None) -> None:

#     if data.target == StoreTarget.FBMEM or data.target == StoreTarget.ZCMEM:
#         weights_arr = cp.asarray(weights.get_inline_allocation()) if weights else None
#         parameters_to_fit_arr = cp.asarray(parameters_to_fit.get_inline_allocation()) if parameters_to_fit else None
#         user_info_arr = cp.asarray(user_info.get_inline_allocation()) if user_info else None
#         parameters, states, chi_squares, number_iterations, execution_time = \
#             gf.fit_gpu(cp.asarray(data.get_inline_allocation()),
#                        weights_arr,
#                        model_id,
#                        cp.asarray(initial_parameters.get_inline_allocation()),
#                        tolerance,
#                        max_number_iterations,
#                        parameters_to_fit_arr,
#                        estimator_id, user_info)
#     else:
#         weights_arr = nnp.asarray(weights.get_inline_allocation()) if weights else None
#         parameters_to_fit_arr = nnp.asarray(parameters_to_fit.get_inline_allocation()) if parameters_to_fit else None
#         user_info_arr = nnp.asarray(user_info.get_inline_allocation()) if user_info else None
#         parameters, states, chi_squares, number_iterations, execution_time = \
#             cf.fit(nnp.asarray(data.get_inline_allocation()),
#                    weights_arr,
#                    model_id,
#                    nnp.asarray(initial_parameters.get_inline_allocation()),
#                    tolerance,
#                    max_number_iterations,
#                    parameters_to_fit_arr,
#                    estimator_id, user_info)

# def fit(data, weights, model_id, initial_parameters, tolerance=None, max_number_iterations=None,
#         parameters_to_fit=None, estimator_id=None, user_info=None):

#     parameters_to_fit_store = get_store(parameters_to_fit) if parameters_to_fit else None
#     user_info_store = get_store(user_info) if user_info else None
#     weights_store = get_store(weights) if weights else None

#     return fit_wrapper(get_store(data),
#                        weights_store,
#                        model_id,
#                        get_store(initial_parameters), get_store(initial_parameters),
#                        tolerance=tolerance,
#                        max_number_iterations=max_number_iterations,
#                        parameters_to_fit=parameters_to_fit_store,
#                        estimator_id=estimator_id,
#                        user_info=user_info_store)

@task(
    variants=("cpu", "gpu"),
    throws_exception=True
)
def fit_wrapper(data: InputStore,
                model_id: int,
                initial_parameters: InputStore, _: OutputStore,
                tolerance: float,
                max_number_iterations: int,
                estimator_id: int,
                parameters: OutputStore,
                states: OutputStore,
                chi_squares: OutputStore,
                number_iterations: OutputStore) -> None:

    if data.target == StoreTarget.FBMEM or data.target == StoreTarget.ZCMEM:
        parameters_output, states_output, chi_squares_output, number_iterations_output, execution_time = \
            gf.fit_gpu(cp.asarray(data.get_inline_allocation()),
                       None,
                       model_id,
                       cp.asarray(initial_parameters.get_inline_allocation()),
                       tolerance,
                       max_number_iterations,
                       None,
                       estimator_id,
                       None)

        parameters_arr = cp.asarray(parameters.get_inline_allocation())
        states_arr = cp.asarray(states.get_inline_allocation())
        chi_squares_arr = cp.asarray(chi_squares.get_inline_allocation())
        number_iterations_arr = cp.asarray(number_iterations.get_inline_allocation())
    else:
        parameters_output, states_output, chi_squares_output, number_iterations_output, execution_time = \
            cf.fit(nnp.asarray(data.get_inline_allocation()),
                   None,
                   model_id,
                   nnp.asarray(initial_parameters.get_inline_allocation()),
                   tolerance,
                   max_number_iterations,
                   None,
                   estimator_id,
                   None)

        parameters_arr = nnp.asarray(parameters.get_inline_allocation())
        states_arr = nnp.asarray(states.get_inline_allocation())
        chi_squares_arr = nnp.asarray(chi_squares.get_inline_allocation())
        number_iterations_arr = nnp.asarray(number_iterations.get_inline_allocation())

    parameters_arr[:] = parameters_output
    states_arr[:] = states_output
    chi_squares_arr[:] = chi_squares_output
    number_iterations_arr[:] = number_iterations_output


def fit(data, weights, model_id, initial_parameters, tolerance=None, max_number_iterations=None,
        parameters_to_fit=None, estimator_id=None, user_info=None,
        parameters=None, states=None, chi_squares=None, number_iterations=None):

    parameters_to_fit_store = get_store(parameters_to_fit) if parameters_to_fit else None
    user_info_store = get_store(user_info) if user_info else None
    weights_store = get_store(weights) if weights else None

    fit_wrapper(get_store(data),
                model_id,
                get_store(initial_parameters), get_store(initial_parameters),
                tolerance=tolerance,
                max_number_iterations=max_number_iterations,
                estimator_id=estimator_id,
                parameters=get_store(parameters),
                states=get_store(states),
                chi_squares=get_store(chi_squares),
                number_iterations=get_store(number_iterations))

if __name__ == '__main__':

    # cuda available checks
    print('CUDA available: {}'.format(gf.cuda_available()))
    if not gf.cuda_available():
        raise RuntimeError(gf.get_last_error())
    print('CUDA versions runtime: {}, driver: {}'.format(*gf.get_cuda_version()))

    # number of fits and fit points
    number_fits = 10000
    size_x = 12
    number_points = size_x * size_x
    number_parameters = 5

    # set input arguments

    # true parameters
    true_parameters = np.array((10, 5.5, 5.5, 3, 10), dtype=np.float32)

    # initialize random number generator
    nnp.random.seed(0)

    # initial parameters (relative randomized, positions relative to width)
    initial_parameters = np.tile(true_parameters, (number_fits, 1))
    initial_parameters[:, (1, 2)] += true_parameters[3] * (-0.2 + 0.4 * np.random.rand(number_fits, 2))
    initial_parameters[:, (0, 3, 4)] *= 0.8 + 0.4 * np.random.rand(number_fits, 3)

    # generate x and y values
    g = np.arange(size_x)
    yi, xi = np.meshgrid(g, g, indexing='ij')
    xi = xi.astype(np.float32)
    yi = yi.astype(np.float32)

    # generate data
    data = generate_gauss_2d(true_parameters, xi, yi)
    data = np.reshape(data, (1, number_points))
    data = np.tile(data, (number_fits, 1))

    # add Poisson noise
    data = np.array(nnp.random.poisson(data))
    data = data.astype(np.float32)

    # tolerance
    tolerance = 0.0001

    # maximum number of iterations
    max_number_iterations = 20

    # estimator ID
    estimator_id = gf.EstimatorID.MLE

    # model ID
    model_id = gf.ModelID.GAUSS_2D

    parameters = np.zeros((number_fits, number_parameters), dtype=np.float32)
    states = np.zeros(number_fits, dtype=np.int32)
    chi_squares = np.zeros(number_fits, dtype=np.float32)
    number_iterations = np.zeros(number_fits, dtype=np.int32)

    # run Gpufit
    # parameters, states, chi_squares, number_iterations, execution_time = fit_gpu(data, None, model_id,
    #                                                                              initial_parameters,
    #                                                                              tolerance,
    #                                                                              max_number_iterations, None,
    #                                                                              estimator_id, None)
    fit(data, None, model_id,
        initial_parameters,
        tolerance,
        max_number_iterations, None,
        estimator_id, None,
        parameters, states, chi_squares, number_iterations)


    # print fit results
    converged = states == 0
    print('*Gpufit*')

    # print summary
    print('\nmodel ID:        {}'.format(model_id))
    print('number of fits:  {}'.format(number_fits))
    print('fit size:        {} x {}'.format(size_x, size_x))
    print('mean chi_square: {:.2f}'.format(np.mean(chi_squares[converged])))
    print('iterations:      {:.2f}'.format(np.mean(number_iterations[converged])))
    # print('time:            {:.2f} s'.format(execution_time))

    # get fit states
    number_converged = np.sum(converged)
    print('\nratio converged         {:6.2f} %'.format(number_converged / number_fits * 100))
    print('ratio max it. exceeded  {:6.2f} %'.format(np.sum(states == 1) / number_fits * 100))
    print('ratio singular hessian  {:6.2f} %'.format(np.sum(states == 2) / number_fits * 100))
    print('ratio neg curvature MLE {:6.2f} %'.format(np.sum(states == 3) / number_fits * 100))

    # mean, std of fitted parameters
    converged_parameters = parameters[converged, :]
    converged_parameters_mean = np.mean(converged_parameters, axis=0)
    converged_parameters_std = np.std(converged_parameters, axis=0)
    print('\nparameters of 2D Gaussian peak')
    for i in range(number_parameters):
        print('p{} true {:6.2f} mean {:6.2f} std {:6.2f}'.format(i, true_parameters[i], converged_parameters_mean[i],
                                                                 converged_parameters_std[i]))
