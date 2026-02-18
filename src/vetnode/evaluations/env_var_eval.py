from __future__ import annotations

import os
from typing import Literal
from pydantic import Field

from vetnode.evaluations.base_eval import BaseEval
from vetnode.evaluations.models import EvalResultStatus


class EnvVarEval(BaseEval):
    """
    Vetnode evaluation that verifies expected environment variables are present
    and (optionally) match expected values.

    Example config concept:
      expected:
        VAR: "value"   # exact string
        SOME_VAR: null # must exist, any value
    """

    name: str
    type: Literal["vetnode.evaluations.env_var_eval.EnvVarEval"]

    # Map of env var -> expected value (str) OR None to mean "must exist"
    expected: dict[str, str | None] = Field(default_factory=dict)

    async def check(self, executor) -> tuple[EvalResultStatus, dict]:
        found: dict[str, str | None] = {}
        missing: list[str] = []
        mismatched: dict[str, dict[str, str | None]] = {}

        for key, value in (self.expected or {}).items():
            actual = os.environ.get(key)
            found[key] = actual

            if actual is None:
                missing.append(key)
                continue

            if value is not None and actual != value:
                mismatched[key] = {"expected": value, "actual": actual}

        ok = (not missing) and (not mismatched)
        status = EvalResultStatus.SUCCESS if ok else EvalResultStatus.FAILED

        metrics = {
            "expected": self.expected,
            "found": found,
            "missing": missing,
            "mismatched": mismatched,
        }
        return status, metrics
