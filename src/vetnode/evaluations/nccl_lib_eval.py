import asyncio
import time
from typing import Literal, Optional
import click
from pydantic import BaseModel
import ctypes, socket
from vetnode.evaluations.base_eval import BaseEval
from vetnode.evaluations.models import BandwidthSize, BinaryByteSize, EvalResultStatus
import numpy as np
import traceback
import cuda.bindings.runtime as cudart

# Define NCCL constants
ncclUniqueId_t = ctypes.c_byte * 128
ncclComm_t = ctypes.c_void_p
cudaStream_t = ctypes.c_void_p

conv_to_GBps = lambda v: v / 10**9


class NCCLEvalWarmUp(BaseModel):
    payload: BinaryByteSize = '256 MB'
    runs: int = 3


def _error_code(err) -> int:
    try:
        return int(err)
    except TypeError:
        return int(err.value)


def _cuda_check(ret, op: str):
    if isinstance(ret, tuple):
        err = ret[0]
        values = ret[1:]
    else:
        err = ret
        values = ()

    if _error_code(err) != 0:
        raise RuntimeError(f"CUDA error during {op}: {err}")

    if len(values) == 0:
        return None
    if len(values) == 1:
        return values[0]
    return values


def _nccl_check(nccl, result: int, op: str):
    if result != 0:
        error_str = nccl.ncclGetErrorString(result)
        message = error_str.decode('utf-8') if error_str else f"error code {result}"
        raise RuntimeError(f"NCCL error during {op}: {message}")


def _send_uid(conn: socket.socket, uid: ncclUniqueId_t):
    # TCP is a byte stream; send() can legally send only a partial UID.
    conn.sendall(bytes(uid))


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    # TCP is a byte stream; recv_into() is not guaranteed to fill the buffer.
    buf = bytearray(nbytes)
    view = memoryview(buf)
    offset = 0
    while offset < nbytes:
        n = sock.recv_into(view[offset:])
        if n == 0:
            raise RuntimeError("socket closed while receiving NCCL unique id")
        offset += n
    return bytes(buf)


def _recv_uid(sock: socket.socket, uid: ncclUniqueId_t):
    raw = _recv_exact(sock, ctypes.sizeof(ncclUniqueId_t))
    ctypes.memmove(ctypes.byref(uid), raw, len(raw))


def _connect_with_retry(host: str, port: int, attempts: int = 30, delay: float = 1.0) -> socket.socket:
    last_error = None
    for _ in range(attempts):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(30)
            s.connect((host, port))
            return s
        except socket.error as e:
            last_error = e
            s.close()
            time.sleep(delay)
    raise RuntimeError(f"failed to connect to NCCL uid server at {host}:{port}: {last_error}")


def _exchange_nccl_ids(nccl, master_node: str, port: int, rank: int, world_size: int, uid, uid_warmup):
    if world_size == 1:
        _nccl_check(nccl, nccl.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
        _nccl_check(nccl, nccl.ncclGetUniqueId(ctypes.byref(uid_warmup)), "ncclGetUniqueId warmup")
        return

    if rank == 0:
        _nccl_check(nccl, nccl.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
        _nccl_check(nccl, nccl.ncclGetUniqueId(ctypes.byref(uid_warmup)), "ncclGetUniqueId warmup")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', port))
            s.settimeout(30)
            s.listen()
            for _ in range(world_size - 1):
                conn, _ = s.accept()
                with conn:
                    _send_uid(conn, uid)
                    _send_uid(conn, uid_warmup)
    else:
        with _connect_with_retry(master_node, port) as s:
            _recv_uid(s, uid)
            _recv_uid(s, uid_warmup)


class NcclLibEval(BaseEval):
    name: str
    type: Literal["vetnode.evaluations.nccl_lib_eval.NcclLibEval"]
    requirements: Literal[["cuda-python", "numpy"]]
    payload: BinaryByteSize = '4 GB'
    method: Literal["allreduce"] = "allreduce"
    topology: Literal["intranode", "internode", "full"] = "full"
    warmup: NCCLEvalWarmUp
    min_bandwidth: BandwidthSize = '15 GB/s'

    def verify(self) -> bool:
        libs = ["libnccl.so"]   # add lib libnccl-net.so "libnvrtc.so"
        for lib in libs:
            try:
                ctypes.CDLL(lib)
            except OSError as e:
                click.echo(f"Could not load {lib}: {e}")
                return False

        if self.context.scheduler == "standalone" or self.context.scheduler is None:
            click.echo("NcclLibEval requires to be run under a supported scheduler (e.g., slurm).")
            return False

        return True

    async def check(self, executor) -> bool:
        try:
            return await asyncio.get_event_loop().run_in_executor(executor, self._check)
        except Exception as e:
            click.echo(f"Error executing check: {e}")
            traceback.print_exc()
            return EvalResultStatus.FAILED, {"error": str(e)}

    def _check(self) -> tuple[Optional[bool], dict]:
        nccl = None
        stream = None
        comm = ncclComm_t()
        warmup_dev_in = None
        warmup_dev_out = None
        dev_in = None
        dev_out = None
        dev_barrier = None

        try:
            master_node = self.context.master_addr
            rank = self.context.rank
            local_rank = self.context.local_rank
            world_size = self.context.world_size

            if self.topology == "internode":
                world_size = self.context.nodes_count
                if self.context.local_rank != 0:
                    return EvalResultStatus.SKIPPED, {"bandwidth": "N/A for non-master ranks in internode topology."}
                rank = int(self.context.rank // self.context.tasks_per_node)

            if self.topology == "intranode":
                world_size = int(self.context.world_size // self.context.nodes_count)
                if rank >= world_size:
                    return EvalResultStatus.SKIPPED, {"bandwidth": "N/A for ranks beyond first node intranode topology."}

            nccl = ctypes.cdll.LoadLibrary('libnccl.so')

            # TODO: re-implement following: https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/pynccl_wrapper.py#L49

            # Define API prototypes
            nccl.ncclGetUniqueId.restype = ctypes.c_int
            nccl.ncclGetUniqueId.argtypes = [ctypes.POINTER(ncclUniqueId_t)]

            nccl.ncclCommInitRank.restype = ctypes.c_int
            nccl.ncclCommInitRank.argtypes = [ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId_t, ctypes.c_int]

            nccl.ncclAllReduce.restype = ctypes.c_int
            nccl.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                           ctypes.c_int, ctypes.c_int, ncclComm_t, cudaStream_t]

            nccl.ncclGetErrorString.restype = ctypes.c_char_p
            nccl.ncclGetErrorString.argtypes = [ctypes.c_int]

            nccl.ncclBroadcast.restype = ctypes.c_int
            nccl.ncclBroadcast.argtypes = [
                ctypes.c_void_p,  # sendbuf
                ctypes.c_void_p,  # recvbuf
                ctypes.c_size_t,  # count
                ctypes.c_int,     # datatype
                ctypes.c_int,     # root
                ncclComm_t,       # comm
                cudaStream_t,     # stream
            ]

            nccl.ncclCommDestroy.restype = ctypes.c_int
            nccl.ncclCommDestroy.argtypes = [ncclComm_t]

            ncclDataType_t = 7  # ncclFloat32
            ncclRedOp_t = 0     # ncclSum

            uid = ncclUniqueId_t()
            uid_warmup = ncclUniqueId_t()
            _exchange_nccl_ids(
                nccl,
                master_node,
                13333 + self.context.eval_id,
                rank,
                world_size,
                uid,
                uid_warmup,
            )

            _cuda_check(cudart.cudaSetDevice(local_rank), "cudaSetDevice")

            stream = _cuda_check(cudart.cudaStreamCreate(), "cudaStreamCreate")
            stream_ptr = ctypes.c_void_p(int(stream))

            comm = ncclComm_t()
            _nccl_check(
                nccl,
                nccl.ncclCommInitRank(ctypes.byref(comm), world_size, uid_warmup, rank),
                "ncclCommInitRank warmup",
            )

            # Warm-up phase
            warmup_n = int(self.warmup.payload) // 4  # np.float32 is 4 bytes
            warmup_host = np.full(warmup_n, rank + 1, dtype=np.float32)
            warmup_dev_in = _cuda_check(cudart.cudaMalloc(warmup_host.nbytes), "cudaMalloc warmup input")
            warmup_dev_out = _cuda_check(cudart.cudaMalloc(warmup_host.nbytes), "cudaMalloc warmup output")
            _cuda_check(
                cudart.cudaMemcpyAsync(
                    warmup_dev_in,
                    warmup_host.ctypes.data,
                    warmup_host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                ),
                "cudaMemcpyAsync warmup input",
            )
            for i in range(self.warmup.runs):
                _nccl_check(
                    nccl,
                    nccl.ncclAllReduce(warmup_dev_in, warmup_dev_out, warmup_n, ncclDataType_t, ncclRedOp_t, comm, stream_ptr),
                    f"warmup ncclAllReduce {i}",
                )

            # Drain warm-up before destroying the warm-up communicator.
            _cuda_check(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize after warmup")

            # Recreate comm to test specific issue with NCCL comm destroy
            _nccl_check(nccl, nccl.ncclCommDestroy(comm), "ncclCommDestroy warmup")
            comm = ncclComm_t()
            _nccl_check(
                nccl,
                nccl.ncclCommInitRank(ctypes.byref(comm), world_size, uid, rank),
                "ncclCommInitRank measurement",
            )

            # Re-warm-up with the measurement communicator using the warm-up-sized buffer.
            for i in range(self.warmup.runs):
                _nccl_check(
                    nccl,
                    nccl.ncclAllReduce(warmup_dev_in, warmup_dev_out, warmup_n, ncclDataType_t, ncclRedOp_t, comm, stream_ptr),
                    f"measurement re-warmup ncclAllReduce {i}",
                )

            # Drain re-warm-up before allocating and timing the measurement buffers.
            _cuda_check(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize after measurement re-warmup")

            _cuda_check(cudart.cudaFree(warmup_dev_in), "cudaFree warmup input")
            warmup_dev_in = None
            _cuda_check(cudart.cudaFree(warmup_dev_out), "cudaFree warmup output")
            warmup_dev_out = None

            # Actual measurement
            n = int(self.payload) // 4  # np.float32 is 4 bytes

            host = np.full(n, rank + 1, dtype=np.float32)
            dev_in = _cuda_check(cudart.cudaMalloc(host.nbytes), "cudaMalloc input")
            dev_out = _cuda_check(cudart.cudaMalloc(host.nbytes), "cudaMalloc output")
            _cuda_check(
                cudart.cudaMemcpyAsync(
                    dev_in,
                    host.ctypes.data,
                    host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                ),
                "cudaMemcpyAsync input",
            )

            # Barrier: align all ranks at the same wall-clock point before the timer.
            # Without this, earlier ranks can include cross-rank wait in the timed op.
            barrier_buf = ctypes.c_float(0.0)
            dev_barrier = _cuda_check(cudart.cudaMalloc(ctypes.sizeof(barrier_buf)), "cudaMalloc barrier")
            _cuda_check(
                cudart.cudaMemcpyAsync(
                    dev_barrier,
                    ctypes.addressof(barrier_buf),
                    ctypes.sizeof(barrier_buf),
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                ),
                "cudaMemcpyAsync barrier input",
            )
            _nccl_check(
                nccl,
                nccl.ncclAllReduce(dev_barrier, dev_barrier, 1, ncclDataType_t, ncclRedOp_t, comm, stream_ptr),
                "pre-timing barrier allreduce",
            )
            _cuda_check(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize before timing")
            _cuda_check(cudart.cudaFree(dev_barrier), "cudaFree barrier")
            dev_barrier = None

            start_time = time.perf_counter()
            _nccl_check(
                nccl,
                nccl.ncclAllReduce(dev_in, dev_out, n, ncclDataType_t, ncclRedOp_t, comm, stream_ptr),
                "timed ncclAllReduce",
            )

            _cuda_check(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize after timing")
            end_time = time.perf_counter()
            elapsedtime = end_time - start_time

            if elapsedtime <= 0:
                return EvalResultStatus.FAILED, {"error": f"Invalid elapsed time: {elapsedtime}"}

            bandwidth = (self.payload / elapsedtime) * (2 * (world_size - 1) / world_size)
            return (
                EvalResultStatus.SUCCESS if bandwidth > self.min_bandwidth else EvalResultStatus.FAILED,
                {"bandwidth": f"{conv_to_GBps(bandwidth):6.2f} GB/s"},
            )

        except Exception as e:
            traceback.print_exc()
            return EvalResultStatus.FAILED, {"error": str(e)}

        finally:
            if nccl is not None and comm is not None and comm.value:
                try:
                    nccl.ncclCommDestroy(comm)
                except Exception:
                    pass

            for ptr in (dev_barrier, dev_in, dev_out, warmup_dev_in, warmup_dev_out):
                if ptr is not None:
                    try:
                        cudart.cudaFree(ptr)
                    except Exception:
                        pass

            if stream is not None:
                try:
                    cudart.cudaStreamDestroy(stream)
                except Exception:
                    pass
