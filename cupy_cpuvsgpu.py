# %%
# !pip install cupy-cuda12x

import numpy as np 
import cupy as cp 
from cupyx.profiler import benchmark

def func(A, device):
    
    if device == 'cpu':
        eigvals, eigvecs = np.linalg.eigh(A)
        
    elif device == 'gpu': 
        eigvals, eigvecs = cp.linalg.eigh(A)

    return eigvals, eigvecs

num_matrices = 10**5
dimension = 110
n_repeat = 10 

A = cp.random.randn(num_matrices, dimension, dimension)
A_cpu = cp.asnumpy(A)

gpu_perf = benchmark(func, args=(A, 'gpu'), n_repeat = n_repeat)
print(gpu_perf)

cpu_perf = benchmark(func, args=(A_cpu, 'cpu'), n_repeat = n_repeat)
print(cpu_perf)

#outputs 
func gpu                 :    CPU: 13166359.937 us   +/- 245.312 (min: 13165908.941 / max: 13166705.959) us     GPU-0: 13166532.422 us   +/- 244.141 (min: 13166081.055 / max: 13166875.977) us
func cpu                :    CPU: 205253646.019 us   +/- 10470069.798 (min: 184576941.039 / max: 215989533.713) us     GPU-0: 205255590.625 us   +/- 10469470.099 (min: 184579250.000 / max: 215988781.250) us 

# CPU DATA 

# root@fi-kermit:/home# lscpu
# Architecture:             x86_64
#   CPU op-mode(s):         32-bit, 64-bit
#   Address sizes:          52 bits physical, 57 bits virtual
#   Byte Order:             Little Endian
# CPU(s):                   224
#   On-line CPU(s) list:    0-223
# Vendor ID:                GenuineIntel
#   Model name:             Intel(R) Xeon(R) Platinum 8480CL
#     CPU family:           6
#     Model:                143
#     Thread(s) per core:   2
#     Core(s) per socket:   56
#     Socket(s):            2
#     Stepping:             7
#     CPU max MHz:          3800.0000
#     CPU min MHz:          800.0000
#     BogoMIPS:             4000.00


# GPU DATA 
# H100 80gb 