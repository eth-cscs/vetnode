


import re
from typing import Any, Dict, List, Optional, Literal, Union
from pydantic import BaseModel, ByteSize
from enum import Enum

class EvalConfiguration(BaseModel, extra='allow'):
    name:str
    type:str
    requirements:Optional[List[str | List[str]]]=None


class SetupResultStatus(Enum):
    SUCCESS = 1
    FAILED = 2
    SKIPPED = 3
    UNKNOWN = 4

    def display(self) -> str:
        return {
            SetupResultStatus.SUCCESS: " 🛠️ ",
            SetupResultStatus.FAILED: " ❌ ",
            SetupResultStatus.SKIPPED: " ⏭️ ",
            SetupResultStatus.UNKNOWN: " ❓ ",
        }[self]

class EvalResultStatus(Enum):
    SUCCESS = 1
    FAILED = 2
    SKIPPED = 3
    UNKNOWN = 4

    def display(self) -> str:
        return {
            EvalResultStatus.SUCCESS: " ✅ ",
            EvalResultStatus.FAILED: " ❌ ",
            EvalResultStatus.SKIPPED: " ⏭️ ",
            EvalResultStatus.UNKNOWN: " ❓ ",
        }[self]


class SetupResult(BaseModel):
   rank:int
   local_rank:int
   eval_id:int
   status:SetupResultStatus=SetupResultStatus.UNKNOWN
   hostname:Optional[str]=None

class EvalResult(BaseModel):
   rank:int
   local_rank:int
   eval_id:int
   eval_name:Optional[str]=None
   eval_type:Optional[str]=None
   status:EvalResultStatus=EvalResultStatus.UNKNOWN
   elapsedtime:Optional[float]=None
   metrics: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
   hostname:Optional[str]=None

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
    hostname:Optional[str]=None

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
