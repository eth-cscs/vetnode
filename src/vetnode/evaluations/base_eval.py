from concurrent.futures import ThreadPoolExecutor
from abc import abstractmethod
import time
from vetnode.evaluations.models import EvalConfiguration, EvalResult, EvalContext, EvalResultStatus



TIMEOUT = 5000
_POOL = ThreadPoolExecutor(max_workers=10)


class BaseEval(EvalConfiguration):

    def __init__(self, context:EvalContext=None, **data) -> None:
        super().__init__(**data)
        self.context = context

    def verify(self)->bool:
        return True
    
    @abstractmethod
    async def check(self, executor:ThreadPoolExecutor)->tuple[EvalResultStatus,dict]:
        pass
    
    async def eval(self)->EvalResult:
        result:EvalResult = EvalResult(**{"rank":self.context.rank, "eval_id":self.context.eval_id, "eval_name":self.name, "eval_type": self.type, "elapsedtime":0, "status":EvalResultStatus.UNKNOWN, "metrics":None})
        
        #async with asyncio.timeout(TIMEOUT):
        start_time = time.time()
        result.status, result.metrics = await self.check(_POOL)
        end_time = time.time()
        result.elapsedtime = end_time-start_time
        return result
