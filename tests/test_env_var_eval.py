import pytest
import textwrap
from click.testing import CliRunner

from vetnode.evaluations.models import EvalContext, EvalResultStatus
from vetnode.evaluations.env_var_eval import EnvVarEval
from vetnode.configuration import Configuration
from vetnode.cli import build_context, load_evals, diagnose


def mk_ctx() -> EvalContext:
    return EvalContext(
        scheduler="standalone",
        rank=0,
        local_rank=0,
        eval_id=0,
        world_size=1,
        master_addr="localhost",
        master_port=29500,
    )


@pytest.mark.asyncio
async def test_must_exist_passes_when_set(monkeypatch):
    monkeypatch.setenv("SOME_FLAG", "")
    ev = EnvVarEval(mk_ctx(), name="env", type="vetnode.evaluations.env_var_eval.EnvVarEval",
                    expected={"SOME_FLAG": None})

    status, metrics = await ev.check(None)
    assert status == EvalResultStatus.SUCCESS
    assert metrics["missing"] == []
    assert metrics["mismatched"] == {}


@pytest.mark.asyncio
async def test_must_exist_fails_when_missing(monkeypatch):
    monkeypatch.delenv("SOME_FLAG", raising=False)
    ev = EnvVarEval(mk_ctx(), name="env", type="vetnode.evaluations.env_var_eval.EnvVarEval",
                    expected={"SOME_FLAG": None})

    status, metrics = await ev.check(None)
    assert status == EvalResultStatus.FAILED
    assert metrics["missing"] == ["SOME_FLAG"]


@pytest.mark.asyncio
async def test_exact_match(monkeypatch):
    monkeypatch.setenv("FI_CXI_RX_MATCH_MODE", "hybrid")
    ev = EnvVarEval(mk_ctx(), name="env", type="vetnode.evaluations.env_var_eval.EnvVarEval",
                    expected={"FI_CXI_RX_MATCH_MODE": "hybrid"})

    status, _ = await ev.check(None)
    assert status == EvalResultStatus.SUCCESS


@pytest.mark.asyncio
async def test_mismatch(monkeypatch):
    monkeypatch.setenv("FI_CXI_RX_MATCH_MODE", "wrong")
    ev = EnvVarEval(mk_ctx(), name="env", type="vetnode.evaluations.env_var_eval.EnvVarEval",
                    expected={"FI_CXI_RX_MATCH_MODE": "hybrid"})

    status, metrics = await ev.check(None)
    assert status == EvalResultStatus.FAILED
    assert "FI_CXI_RX_MATCH_MODE" in metrics["mismatched"]


def test_yaml_populates_expected_env(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(textwrap.dedent("""
        name: test-config
        scheduler: standalone
        evals:
          - name: env-check
            type: vetnode.evaluations.env_var_eval.EnvVarEval
            expected:
              SOME_FLAG: null
              FI_CXI_RX_MATCH_MODE: hybrid
    """).lstrip())

    # Make Configuration read this YAML
    Configuration._yaml_file = str(cfg)
    conf = Configuration()

    ctx = build_context(conf)
    evals = load_evals(ctx, conf.evals)

    assert len(evals) == 1
    ev = evals[0]

    assert ev.expected["SOME_FLAG"] is None
    assert ev.expected["FI_CXI_RX_MATCH_MODE"] == "hybrid"


@pytest.mark.parametrize(
    "env, expected_exit, expected_token",
    [
        ({"FI_CXI_RX_MATCH_MODE": "hybrid", "FI_MR_CACHE_MONITOR": "userfaultfd"}, 0, "Vetted:"),
        ({"FI_CXI_RX_MATCH_MODE": "nope",   "FI_MR_CACHE_MONITOR": "userfaultfd"}, 1, "Cordon:"),
        ({}, 1, "Cordon:"),
    ],
)
def test_diagnose_env_eval(tmp_path, monkeypatch, env, expected_exit, expected_token):
    # Set/clear relevant env vars
    for k in ["FI_CXI_RX_MATCH_MODE", "FI_MR_CACHE_MONITOR"]:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(textwrap.dedent("""
        name: test-config
        scheduler: standalone
        evals:
          - name: env-check
            type: vetnode.evaluations.env_var_eval.EnvVarEval
            expected:
              FI_CXI_RX_MATCH_MODE: hybrid
              FI_MR_CACHE_MONITOR: userfaultfd
    """).lstrip())

    runner = CliRunner()
    result = runner.invoke(diagnose, [str(cfg)])

    print(result.output)
    assert result.exit_code == expected_exit
    assert expected_token in result.output
