
import asyncio
import os
from typing import Literal

from vetnode.evaluations.models import EvalResultStatus
from vetnode.evaluations.base_eval import BaseEval
from ctypes import CDLL

from cuda.bindings import driver, nvrtc
import numpy as np


def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))
    
saxpy = """\
extern "C" __global__
void saxpy(float a, float *x, float *y, float *out, size_t n)
{
 size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
 if (tid < n) {
   out[tid] = a * x[tid] + y[tid];
 }
}
"""

class CUDAEval(BaseEval):
    name:str
    type: Literal["vetnode.evaluations.cuda_eval.CUDAEval"]
    requirements: Literal[["cuda-python==12.*","numpy"],["cuda-python==13.*","numpy"]]
    cuda_home: str

    def verify(self)->bool:
        libc = CDLL(f"{self.cuda_home}/lib64/libnvrtc.so")
        if libc is None:
            return False
        for filename in os.listdir(f"{self.cuda_home}/lib64"):
            if filename.endswith(".so"):
                lib_path = os.path.join(f"{self.cuda_home}/lib64", filename)
                CDLL(f"{lib_path}")
        return True


    async def check(self,executor)->tuple[bool,dict]:
        return await asyncio.get_event_loop().run_in_executor(executor, self._check)


    def _check(self)->tuple[bool,dict]:

        (err,) = driver.cuInit(0)
        self.checkCudaErrors(err)

        err, drv_version = driver.cuDriverGetVersion()

        err, nvrtc_major, nvrtc_minor = nvrtc.nvrtcVersion()

        err, ndevice = driver.cuDeviceGetCount()
        self.checkCudaErrors(err)

        err, cuDevice = driver.cuDeviceGet(0)
        self.checkCudaErrors(err)


        err, major = driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice)
        self.checkCudaErrors(err)
        err, minor = driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice)
        self.checkCudaErrors(err)

        arch_arg = bytes(f"--gpu-architecture=compute_{major}{minor}", "ascii")
        err, prog =nvrtc.nvrtcCreateProgram(str.encode(saxpy), b"saxpy.cu", 0, [], [])
        self.checkCudaErrors(err)

        opts = [
                b"--fmad=false",
                #b'--ptx-version=8.1',  # Explicitly set PTX version
                arch_arg
        ]

        (err, )=nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        self.checkCudaErrors(err)

        # Get PTX from compilation
        err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
        self.checkCudaErrors(err)
        ptx = b"\x00" * ptxSize
        (err, )= nvrtc.nvrtcGetPTX(prog, ptx)
        self.checkCudaErrors(err)

        err, version = driver.cuDriverGetVersion()
        self.checkCudaErrors(err)
        # Create context
        if version >= 1300:
            ctxParams = driver.CUctxCreateParams()  # Default initialized
            err, context = driver.cuCtxCreate(ctxParams,0, cuDevice)
        else:
            err, context = driver.cuCtxCreate(0, cuDevice)
        
        self.checkCudaErrors(err)
        
        err, = driver.cuCtxSetCurrent(context)
        self.checkCudaErrors(err)

        # Load PTX as module data and retrieve function
        #ptx = np.char.array(ptx)
        # Note: Incompatible --gpu-architecture would be detected here
        err,module = driver.cuModuleLoadData(ptx)
        self.checkCudaErrors(err)
        err, kernel = driver.cuModuleGetFunction(module, b"saxpy")
        self.checkCudaErrors(err)

        NUM_THREADS = 512
        NUM_BLOCKS = 32768

        # Scalar values
        a_host = np.array([2.0], dtype=np.float32)
        n = np.array(NUM_THREADS * NUM_BLOCKS, dtype=np.uint32)
        bufferSize = n * a_host.itemsize

        # Host arrays
        hX = np.random.rand(n).astype(np.float32)
        hY = np.random.rand(n).astype(np.float32)
        hOut = np.zeros(n, dtype=np.float32)

        # Allocate device memory
        err, dX = driver.cuMemAlloc(bufferSize)
        err, dY = driver.cuMemAlloc(bufferSize)
        err, dOut = driver.cuMemAlloc(bufferSize)

        # Create stream
        err, stream = driver.cuStreamCreate(0)

        # Copy data to device
        err, = driver.cuMemcpyHtoDAsync(dX, hX.ctypes.data, bufferSize, stream)
        err, = driver.cuMemcpyHtoDAsync(dY, hY.ctypes.data, bufferSize, stream)

        # Setup kernel arguments as array of pointers
        # Each element points to the actual argument value
        dX_ptr = np.array([int(dX)], dtype=np.uint64)
        dY_ptr = np.array([int(dY)], dtype=np.uint64)
        dOut_ptr = np.array([int(dOut)], dtype=np.uint64)

        # Array of pointers to arguments
        args = [a_host, dX_ptr, dY_ptr, dOut_ptr, n]
        args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

        # Launch kernel
        err, = driver.cuLaunchKernel(
            kernel,
            NUM_BLOCKS,  # grid x dim
            1,  # grid y dim
            1,  # grid z dim
            NUM_THREADS,  # block x dim
            1,  # block y dim
            1,  # block z dim
            0,  # dynamic shared memory
            stream,  # stream
            args.ctypes.data,  # kernel arguments
            0,  # extra (ignore)
        )
        self.checkCudaErrors(err)

        driver.cuMemcpyDtoHAsync(hOut.ctypes.data, dOut, bufferSize, stream)
        driver.cuStreamSynchronize(stream)
        
        hZ = a_host * hX + hY


        if not np.allclose(hOut, hZ):
            return EvalResultStatus.FAILED,{"error":"outside tolerance for host-device vectors","cuda_version":drv_version, "NVRTC_version": f"{nvrtc_major}.{nvrtc_minor}"}
        
        driver.cuStreamDestroy(stream)
        driver.cuMemFree(dX)
        driver.cuMemFree(dY)
        driver.cuMemFree(dOut)
        driver.cuModuleUnload(module)
        driver.cuCtxDestroy(context)

        return EvalResultStatus.SUCCESS,{"cuda_version":f"{drv_version}", "NVRTC_version": f"{nvrtc_major}.{nvrtc_minor}"}

    def checkCudaErrors(self, err):
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError("CUDA error code={}({})".format(err, _cudaGetErrorEnum(err)))
        