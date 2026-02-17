import asyncio
from typing import List
import click
import traceback
import socket  
from vetnode.configuration import Configuration
from vetnode.evaluations.models import EvalConfiguration,EvalContext,EvalResult,EvalResultStatus, SetupResult, SetupResultStatus, SetupResultStatus
import os
from vetnode.commands.scontrol.scontrol_command import ScontrolCommand
import struct
import sys
import subprocess
from pydoc import locate



def build_context(configuration:Configuration)->EvalContext:
    main_context:EvalContext    = EvalContext()
    main_context.scheduler = configuration.scheduler
    match configuration.scheduler:
            case "slurm":
                main_context.rank=int(os.environ["SLURM_PROCID"])
                main_context.local_rank = int(os.environ["SLURM_LOCALID"])
                main_context.nodes = asyncio.run(ScontrolCommand().run()).hostnames
                main_context.master_addr = main_context.nodes[0]
                main_context.master_port = 29500 #Default port used to collect evaluation results
                main_context.world_size = int(os.environ['SLURM_NTASKS'])
                main_context.nodes_count = int(os.environ['SLURM_JOB_NUM_NODES'])
                main_context.tasks_per_node = int(main_context.world_size/main_context.nodes_count)
            case "standalone":
                main_context.rank=0
                main_context.local_rank = 0
                main_context.nodes = None
                main_context.master_addr = "localhost"
                main_context.master_port = 29500
                main_context.world_size = 1
                main_context.nodes_count = None
                main_context.tasks_per_node = None
            case _:
                raise NotImplementedError("Support for the rquested scheduler has not been implemented.")
    return main_context

@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--skip-install", default=False, is_flag=True, help="Skip installation of the evals requirements. Useful when requirements are already installed (see setup command).")
@click.option("--verbose", default=False, is_flag=True, help="Enable verbose output.")
def diagnose(config,skip_install,verbose) -> None:
    hostname:str = socket.gethostname()
    Configuration._yaml_file = config
    configuration = Configuration()
    main_context= build_context(configuration)
    
    evals = load_evals(main_context, configuration.evals)
    processes = asyncio.run(run_evals(main_context,evals,skip_install=skip_install,index_url=configuration.pip.index_url))
    healthy:bool=True
    for results in processes:
        if isinstance(results, Exception):
            click.secho(f"Node: {hostname} \t unexpected exception: {results}", fg='red')
            traceback.print_tb(results.__traceback__)
            healthy=False
            continue
        if isinstance(results, List):
            for result in results:
                if isinstance(result, Exception):
                    click.secho(f"Node: {hostname} \t unexpected exception: {result}", fg='red')
                    traceback.print_tb(result.__traceback__)
                    healthy=False
                else:
                    if not result.status == EvalResultStatus.SUCCESS and not result.status == EvalResultStatus.SKIPPED:
                        healthy=False
                    if verbose:
                        click.secho(f"Node: {hostname} \t result:{result}", fg='green' if result.status == EvalResultStatus.SUCCESS else 'red')
            continue
        click.secho(f"Node: {hostname} \t result:{results}", fg='red')

    #if healthy:
    #    click.echo(f"Vetted: {hostname}")
    #else:
    #    click.echo(f"Cordon: {hostname}")
    #    sys.exit(1)

@click.command()
@click.argument("config", type=click.Path(exists=True))
def setup(config) -> None:
    Configuration._yaml_file = config
    configuration = Configuration()
    main_context= build_context(configuration)
    load_evals(main_context,configuration.evals, install=True, index_url=configuration.pip.index_url)


async def send_int(writer, value: int):
    writer.write(struct.pack("!i", value))
    await writer.drain()

async def recv_int(reader) -> int:
    data = await reader.readexactly(4)
    return struct.unpack("!i", data)[0]

async def send_str(writer, msg: str):
    data = msg.encode()
    await send_int(writer, len(data))
    writer.write(data)
    await writer.drain()

async def recv_str(reader) -> str:
    length = await recv_int(reader)
    data = await reader.readexactly(length)
    return data.decode()


async def run_evals(main_context,evals,skip_install:bool=True,index_url: str = None):
    tasks = []
    if main_context.rank==0 and main_context.local_rank==0:
        tasks.append(asyncio.create_task(synchronize_workers(main_context,evals)))
    
    tasks.append(asyncio.create_task(run_evals_worker(main_context,evals,skip_install=skip_install,index_url=index_url)))
    return await asyncio.gather(*tasks, return_exceptions=True)





async def run_evals_worker(main_context,evals, skip_install:bool=True,index_url: str = None):
    results = []
    for attempt in range(10):
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(main_context.master_addr, main_context.master_port), timeout=5.0)
            try:
                while True:
                    instruction = await recv_str(reader)
                    if instruction == "STOP":
                        return results
                    
                    if instruction.startswith("SETUP"):
                        _, eval_id_str = instruction.split(":")
                        eval_id = int(eval_id_str)
                        result = SetupResult(rank=main_context.rank, eval_id=eval_id)
                        try:
                            if not skip_install and main_context.local_rank==0 and evals[eval_id].requirements:
                                load_requirements(evals[eval_id].requirements,index_url)
                                result.status = SetupResultStatus.SUCCESS
                            else:
                                result.status = SetupResultStatus.SKIPPED
                        except Exception as ex:
                            click.secho(f"Skipped: {evals[eval_id].name} (error: {ex})", fg='red')
                            result.status = SetupResultStatus.FAILED
                        finally:
                            await send_str(writer, f"{result.model_dump_json()}")

                    if instruction.startswith("EVAL"):
                        _, eval_id_str = instruction.split(":")
                        eval_id = int(eval_id_str)
                        eval = evals[eval_id]
                        result = EvalResult(rank=main_context.rank, eval_id=eval_id)
                        try:
                            if eval.verify():
                                result = await eval.eval()
                                results.append(result)
                        except Exception as ex:
                            click.secho(f"Skipped: {eval.name} (error: {ex})", fg='red')
                        finally:
                            await send_str(writer, f"{result.model_dump_json()}")
            finally:
                writer.close()
                await writer.wait_closed()
                return results
        except (asyncio.TimeoutError, ConnectionRefusedError) as e:
            await asyncio.sleep(2.0)
            continue
        except Exception as e:
            raise e
    raise ConnectionError(
        f"Failed to connect to master node {main_context.master_addr}:{main_context.master_port} after {attempt} attempts"
    )


async def synchronize_workers(main_context,evals):
    clients = []
    results = [[None] * main_context.world_size  for _ in range(len(evals))]

    async def handle_client(reader, writer):
        try:
            event = asyncio.Event()
            clients.append((reader, writer, event))
            click.secho(" 🚀 ", fg='green', nl=False)
            await event.wait()  # keep handler alive
        except Exception as e:
            raise e
        finally:
            writer.close()
            await writer.wait_closed()


    server = await asyncio.start_server(handle_client, '0.0.0.0', main_context.master_port)
    async with server:
        try:
            click.secho("Igniting workers: ", fg='green', nl=False)
            while len(clients) < main_context.world_size:
                await asyncio.sleep(0.2)
            click.echo("")
            for i in range(len(evals)):

                # Send task index
                click.secho(f"Setting up {evals[i].name}:", fg='blue',nl=False)
                for _, writer,_ in clients:
                    await send_str(writer, f"SETUP:{i}")

                
                for reader, _, _ in clients:
                    setup_json = await recv_str(reader)
                    try:
                        result = SetupResult.model_validate_json(setup_json)
                    except Exception as e:
                        click.secho(f"Error deserializing setup result JSON: {e}", fg='red')
                        continue
                    match result.status:
                        case SetupResultStatus.SUCCESS:
                            click.secho(f" 🛠️[rank: {result.rank}] ", fg='green', nl=False)
                        case SetupResultStatus.SKIPPED:
                            continue
                        case SetupResultStatus.FAILED:
                            click.secho(" ❌ ", fg='red', nl=False)
                        case _:
                            click.secho(" ❓ ", fg='red', nl=False)
                click.echo("")
                   
            for i in range(len(evals)):

                # Send task index
                click.secho(f"Evaluating {evals[i].name}:", fg='blue',nl=False)
                for _, writer,_ in clients:
                    await send_str(writer, f"EVAL:{i}")

                
                for reader, _, _ in clients:
                    result_json = await recv_str(reader)
                    try:
                        result = EvalResult.model_validate_json(result_json)
                    except Exception as e:
                        click.secho(f"Error validating result JSON: {e}", fg='red')
                        continue
                    match result.status:
                        case EvalResultStatus.SUCCESS:
                            click.secho(" ✅ ", fg='green', nl=False)
                        case EvalResultStatus.FAILED:
                            click.secho(" ❌ ", fg='red', nl=False)
                        case EvalResultStatus.SKIPPED:
                            click.secho(" ⏭️ ", fg='blue', nl=False)
                        case _:
                            click.secho(" ❓ ", fg='red', nl=False)
                    results[result.eval_id][result.rank] = result
                click.echo("")         
        except Exception as e:
            raise e
        finally:
            for _, writer, _ in clients:
                await send_str(writer, "STOP")
            for _, _, event in clients:
                event.set()
            server.close()  # Stop accepting new connections
            await server.wait_closed() 
        


def load_evals( main_context:EvalContext, eval_configs: List[EvalConfiguration], install:bool=False,index_url: str = None):
    evals = []
    for i,eval in enumerate(eval_configs):
        eval_context = main_context.model_copy(update={'eval_id': i})
        #Load class dynamically
        try:
            if install and eval.requirements:
                load_requirements(eval.requirements,index_url)
            eval_class = locate(eval.type)
            evals.append(eval_class(eval_context,**eval.model_dump()))
        except Exception as ex:
            click.secho(f"Skipped: {eval.name} (error: {ex})", fg='red')
    return evals

def load_requirements(requirements: List[str], index_url: str = None):
    for package in requirements:
        cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir","-q"]
        if index_url:
            cmd += ["--index-url",index_url]
        if isinstance(package, str):
            cmd.append(package)
        else:
            cmd += package
        subprocess.check_call(cmd)