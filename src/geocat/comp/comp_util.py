import numpy as np
import xarray as xr
import dask.array as da


def _is_duck_array(value, xp):
    """Returns True when ``value`` is array-like."""
    if isinstance(value, xp.ndarray):
        return True
    return (hasattr(value, "ndim") and hasattr(value, "shape") and
            hasattr(value, "dtype") and hasattr(value, "__array_function__") and
            hasattr(value, "__array_ufunc__"))


def _import_cupy():
    """imports the cupy and checks if not installed."""
    try:
        import cupy as cp
        return cp
    except ImportError as e:
        print(f"Cupy is not installed for GPU computation!")
        pass  # module doesn't exist, deal with it. """


def _convert_to_gpu_array(inputs):
    xp = _import_cupy()
    inputs_gpu = []
    in_types = [type(item) for item in inputs]
    if xp.ndarray in in_types:
        #if the inputs are already cupy arrays
        return inputs
    elif np.ndarray in in_types:
        #convert numpy to cupy
        for item in inputs:
            inputs_gpu.append(xp.asarray(item))
    elif xr.DataArray in in_types:
        #convert xarray
        in_types = [type(item.data) for item in inputs]
        if np.ndarray in in_types:
            #xarray with type(item.data) = numpy.ndarray
            for item in inputs:
                inputs_gpu.append(xr.DataArray(xp.asarray(item.data)))
        elif da.Array in in_types:
            #xarray with type(item.data) = dask array
            for item in inputs:
                inputs_gpu.append(xr.DataArray(item.data.map_blocks(
                    xp.asarray)))
        else:
            inputs_gpu = inputs
    elif da.Array in in_types:
        #convert dask array
        for item in inputs:
            inputs_gpu.append(item.map_blocks(xp.asarray))
    else:
        return inputs
    return inputs_gpu


def _convert_to_cpu_array(inputs):
    cp = _import_cupy()
    inputs_cpu = []
    in_types = [type(item) for item in inputs]
    if cp.ndarray in in_types:
        #convert cupy to numpy
        for item in inputs:
            inputs_cpu.append(cp.asnumpy(item))
    elif xr.DataArray in in_types:
        #convert the xarray
        in_types = [type(item.data) for item in inputs]
        if cp.ndarray in in_types:
            #xarray with type(item.data) = numpy.ndarray
            for item in inputs:
                inputs_cpu.append(xr.DataArray(cp.asnumpy(item.data)))
        elif da.Array in in_types:
            #xarray with type(item.data) = dask array
            for item in inputs:
                inputs_cpu.append(xr.DataArray(item.data.map_blocks(
                    cp.asnumpy)))
        else:
            inputs_cpu = inputs
    elif da.Array in in_types:
        for item in inputs:
            #convert dask array
            inputs_cpu.append(item.map_blocks(cp.asnumpy))
    else:
        return inputs
    return inputs_cpu
