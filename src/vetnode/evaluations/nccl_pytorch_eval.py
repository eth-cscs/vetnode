
import asyncio
import datetime
from typing import Literal
import numpy as np 
from pydantic import BaseModel
import click

from vetnode.commands.scontrol.scontrol_command import ScontrolCommand
from vetnode.evaluations.base_eval import BaseEval
import torch
import torch.distributed as dist
from vetnode.evaluations.models import BandwidthSize, BinaryByteSize,EvalResultStatus

# following the common networking hw spec convention which uses base 10, instead of 2 for bps/Bps (it makes speed look bigger than it is)
conv_to_GBps = lambda v : v/10**9




class NCCLEvalWarmUp(BaseModel):
    payload:BinaryByteSize= '256 MB'
    runs:int= 3

class NcclPytorchEval(BaseEval):
    name:str
    type: Literal["vetnode.evaluations.nccl_pytorch_eval.NcclPytorchEval"]
    requirements: Literal[[['torch','--index-url','https://download.pytorch.org/whl/cu129'],"numpy"],[['torch','--index-url','https://download.pytorch.org/whl/cu130'],"numpy"],[['torch','--index-url','https://download.pytorch.org/whl/nightly/rocm7.1'],"numpy"]]
    scheduler:  Literal["slurm","openPBS"]
    payload: BinaryByteSize = '4 GB'
    method: Literal["broadcast","roundrobin","allreduce","gather","allreduce-gather"] = "broadcast"
    warmup: NCCLEvalWarmUp
    min_bandwidth: BandwidthSize = '15 GB/s'
    
    def verify(self)->bool:
        if self.context.scheduler == "standalone" or self.context.scheduler is None:
            click.echo("NcclPytorchEval requires to be run under a supported scheduler (e.g., slurm).")
            return False
        
        return True

    async def check(self,executor)->bool:
        return await asyncio.get_event_loop().run_in_executor(executor, self._check)


    def _check(self)->tuple[bool,dict]:

        master_node = self.context.master_addr
        rank = self.context.rank
        local_rank = self.context.local_rank
        world_size = self.context.world_size

        dist.init_process_group(
            backend="nccl",
            init_method="tcp://{}:{}".format(master_node, 6011+self.context.eval_id),
            timeout=datetime.timedelta(seconds=30),
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(local_rank)
        
        tensor = None
        # /4 is for 4 bytes in fp32
        tensor = torch.rand(self.warmup.payload//4, 1, dtype=torch.float32).cuda(local_rank)
        for i in range(self.warmup.runs):
             self.timed_allreduce(local_rank,rank,tensor,self.warmup.payload,world_size)

        # /4 is for 4 bytes in fp32
        tensor = torch.rand(self.payload//4, 1, dtype=torch.float32).cuda(local_rank)
        mesurment_matrix = []
        
        match self.method:
            case "allreduce":
                bandwidth = self.timed_allreduce(local_rank,rank,tensor,self.payload,world_size)
            case "roundrobin":
                bandwidth,mesurment_matrix = self.timed_roundrobin(local_rank,rank,tensor,self.payload,world_size)
            case "broadcast":
                bandwidth = self.timed_broadcast(local_rank,rank,tensor,self.payload,world_size)
            case "gather":
                bandwidth = self.timed_gather(local_rank,rank,tensor,self.payload,world_size)
            case "allreduce-gather":
                bandwidth = self.timed_allreduce_gather(local_rank,rank,tensor,self.payload,world_size)
            case _:
                raise NotImplementedError("Bandwidth test method not implemented.")
        
        dist.destroy_process_group()
        
        return EvalResultStatus.SUCCESS if bandwidth > self.min_bandwidth else EvalResultStatus.FAILED, {"bandwidth":f"{conv_to_GBps(bandwidth):6.2f} GB/s"}
    
    def timed_allreduce_gather(self,local_rank,rank,tensor,size,ranks):
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        tensor_b = tensor.detach().clone()

        dist.all_reduce(tensor_b,op=dist.ReduceOp.SUM)

        start_event.record()
        if rank == 0:
            gather_list = [torch.zeros_like(tensor) for _ in range(ranks)]
        else:
            gather_list = None

        dist.gather(tensor, gather_list, dst=0)

        end_event.record()

        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        duration = start_event.elapsed_time(end_event) / 1000
        bandwidth = size/duration        
        return bandwidth * (ranks - 1)
    
    def timed_gather(self,local_rank,rank,tensor,size,ranks):
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        dist.barrier(device_ids=[local_rank])

        start_event.record()
        if rank == 0:
            gather_list = [torch.zeros_like(tensor) for _ in range(ranks)]
        else:
            gather_list = None

        dist.gather(tensor, gather_list, dst=0)

        end_event.record()
        torch.cuda.synchronize()
        duration = start_event.elapsed_time(end_event) / 1000
        bandwidth = size/duration        
        return bandwidth * (ranks - 1)

    
    def timed_allreduce(self,local_rank,rank,tensor,size,ranks):
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        dist.barrier(device_ids=[local_rank])
        start_event.record()
        dist.all_reduce(tensor)
        end_event.record()
        torch.cuda.synchronize()
        duration = start_event.elapsed_time(end_event) / 1000
        bandwidth = size/duration        
        return bandwidth * (2*(ranks - 1) / ranks)
    
    def timed_roundrobin(self,local_rank,rank,tensor,size,ranks):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        measurments = np.array([])
        metadata = []
        for i in range(ranks):
            for j in [j for j in range(ranks) if j != i]:

                #All processes wait here    
                dist.barrier(device_ids=[local_rank])
                
                if rank == i or rank == j:
                    start_event.record()
                    if rank == i:
                        dist.send(tensor=tensor, dst=j)
                    else:
                        dist.recv(tensor=tensor, src=i)
                    end_event.record()
                    torch.cuda.synchronize()
                    duration = start_event.elapsed_time(end_event) / 1000
                    if rank == i:
                        measurments = np.append(measurments, size/duration )
                        metadata.append({"from":i, "to":j, "bandwidth":f"{conv_to_GBps(size/duration):6.2f} GB/s"})     
        return np.min(measurments),metadata 

    def timed_broadcast(self,local_rank,rank,tensor,size,ranks):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        bandwidth = 0
        for i in range(ranks):
                #All processes wait here    
                dist.barrier(device_ids=[local_rank])
                start_event.record()
                dist.broadcast(tensor, i)
                end_event.record()
                torch.cuda.synchronize()
                duration = start_event.elapsed_time(end_event) / 1000
                if rank == i:
                    bandwidth= size/duration  
        return bandwidth * (ranks - 1)


