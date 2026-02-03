
import asyncio
from typing import Literal
import random
from vetnode.evaluations.base_eval import BaseEval
from vetnode.evaluations.models import EvalResultStatus


class RandomEval(BaseEval):
    name:str
    type: Literal["vetnode.evaluations.random.RandomEval"]

    async def check(self,executor)->tuple[bool,dict]:
        return await asyncio.get_event_loop().run_in_executor(executor, random.choice, [EvalResultStatus.SUCCESS, EvalResultStatus.FAILED] ), None