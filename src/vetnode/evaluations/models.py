


import re
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, ByteSize
from enum import Enum

class EvalConfiguration(BaseModel, extra='allow'):
    name:str
    type:str
    requirements:Optional[List[str | List[str]]]=None


class EvalResultStatus(Enum):
    SUCCESS = 1
    FAILED = 2
    SKIPPED = 3
    UNKNOWN = 4


class EvalResult(BaseModel):
   rank:int
   eval_id:int
   eval_name:Optional[str]=None
   eval_type:Optional[str]=None
   status:EvalResultStatus=EvalResultStatus.UNKNOWN
   elapsedtime:Optional[float]=None
   metrics:Optional[Dict[str,str]]=None

class EvalContext(BaseModel):
    eval_id:int=None
    nodes:Optional[List[str]]=None
    local_rank:Optional[int]=None
    rank:Optional[int]=None
    world_size:Optional[int]=None
    nodes_count:Optional[int]=None
    tasks_per_node:Optional[int]=None
    master_addr:Optional[str]=None
    master_port:Optional[int]=None
    scheduler:Literal["slurm", "standalone"]="slurm"

class BinaryByteSize(ByteSize):
    byte_sizes = {
        'b': 1,
        'kb': 2**10,
        'mb': 2**20,
        'gb': 2**30,
        'tb': 2**40,
        'pb': 2**50,
        'eb': 2**60,
    }

class BandwidthSize(ByteSize):
    byte_sizes = {
        'b/s': 1,
        'kb/s': 2**10,
        'mb/s': 2**20,
        'gb/s': 2**30,
        'tb/s': 2**40,
        'pb/s': 2**50,
        'eb/s': 2**60,
    }
    byte_string_pattern = r'^\s*(\d*\.?\d+)\s*([\w\/]+)?'
    byte_string_re = re.compile(byte_string_pattern, re.IGNORECASE)