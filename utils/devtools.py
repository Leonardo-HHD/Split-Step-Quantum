import numpy as np
def num_check(var):
    if isinstance(var, np.ndarray):
        print(f"Data shape: {var.shape}, dtype: {var.dtype}, size: {var.size}*{var.itemsize} bytes = {var.nbytes/1024/1024} MB")
    if np.isinf(var).any():
        inf_indices = np.where(np.isinf(var))
        print(f"Number of Inf components: {len(inf_indices[0])}")
        raise ValueError(f"Inf appears in <{hex(id(var))}>")
    if np.isnan(var).any():
        nan_indices = np.where(np.isnan(var))
        print(f"Number of NaN components: {len(nan_indices[0])}")
        raise ValueError(f"NaN appears in <{hex(id(var))}>")