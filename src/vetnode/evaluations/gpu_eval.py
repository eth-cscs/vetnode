from typing import Literal

from vetnode.commands.nvidiasmi.models import NvidiaSMIOutput
from vetnode.commands.nvidiasmi.nvidia_smi_command import NvidiaSMICommand
from vetnode.evaluations.base_eval import BaseEval
from vetnode.evaluations.models import EvalResultStatus

class GPUEval(BaseEval):
    name:str
    type: Literal["vetnode.evaluations.gpu_eval.GPUEval"]
    max_temp: int
    max_used_memory: float

    async def check(self,executor)->tuple[bool,dict]:
        gpu_info : NvidiaSMIOutput = await NvidiaSMICommand().run()
        
        if len([gpu for gpu in gpu_info.gpus if gpu.temp >= self.max_temp or (gpu.memory_used / gpu.memory_total) > self.max_used_memory ]) == 0:
            result = EvalResultStatus.SUCCESS
        else:
            result = EvalResultStatus.FAILED
        
        return result, gpu_info.model_dump()

