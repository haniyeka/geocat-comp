import numpy as np
import xarray as xr
import dask.array as da

def _is_duck_array(value):
    """Returns True when ``value`` is array-like."""
    if isinstance(value, np.ndarray):
        return True
    return (hasattr(value, "ndim") and hasattr(value, "shape") and
            hasattr(value, "dtype") and hasattr(value, "__array_function__") and
            hasattr(value, "__array_ufunc__"))

def _import_cupy():
    """imports the cupy and checks if not installed"""
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
    #convert the numpy to cupy
    if np.ndarray in in_types:
        for item in inputs: 
            inputs_gpu.append(xp.asarray(item))
    #convert the xarray
    elif xr.DataArray in in_types:
        in_types = [type(item.data) for item in inputs]
        if np.ndarray in in_types:
            for item in inputs: 
                inputs_gpu.append(xr.DataArray(xp.asarray(item.data)))
        elif da.Array in in_types:
            for item in inputs: 
                inputs_gpu.append(xr.DataArray(item.data.map_blocks(xp.asarray)))
    return inputs_gpu

def _convert_to_cpu_array(inputs):
    cp = _import_cupy()
    inputs_gpu = []
    in_types = [type(item) for item in inputs]
    #convert the numpy to cupy
    if cp.ndarray in in_types:
        for item in inputs: 
            inputs_gpu.append(cp.asnumpy(item))
    #convert the xarray
    elif xr.DataArray in in_types:
        in_types = [type(item.data) for item in inputs]
        if cp.ndarray in in_types:
            for item in inputs: 
                inputs_gpu.append(xr.DataArray(cp.asnumpy(item.data)))
        elif da.Array in in_types:
            for item in inputs: 
                inputs_gpu.append(xr.DataArray(item.data.map_blocks(cp.asnumpy)))
    return inputs_gpu