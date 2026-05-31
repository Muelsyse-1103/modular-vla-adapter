# Remote Environment Process

The environment process is separate from the model process. This keeps simulator
dependencies such as LIBERO, robosuite, MuJoCo, and CALVIN out of the Qwen/ViT
training environment.

```text
Model / runtime process
  - prismatic_adapter
  - torch / transformers / timm
  - VLAAdapter policy
        |
        | ZMQ REQ/REP
        v
Environment process
  - env_process
  - LIBERO / CALVIN / fake backend
  - reset / step / render / success check
```

## Packages

```text
env_process/
├── protocols.py          # HELLO / LIST_TASKS / RESET / STEP / RENDER / CLOSE
├── codecs.py             # numpy array encoding for obs/proprio/images
├── backends/
│   ├── base.py           # EnvBackend contract
│   └── fake.py           # dependency-light smoke backend
└── clients/
    └── zmq_server.py     # synchronous ZMQ REP server

vla_runtime/
├── env_client.py         # ZMQ REQ client
├── policies/             # runtime policy wrappers
├── rollouts/             # action queue rollout loop
├── runners/              # evaluation runner
├── buffers/              # episode containers
└── recorder.py           # JSONL/metrics writer
```

## Smoke Test

Terminal A:

```bash
python scripts/serve_fake_env.py --endpoint tcp://127.0.0.1:5555
```

Terminal B:

```bash
python scripts/eval_with_remote_env.py \
  --endpoint tcp://127.0.0.1:5555 \
  --output-dir outputs/remote_eval_smoke
```

This writes:

```text
outputs/remote_eval_smoke/
├── episodes.jsonl
└── metrics.json
```

## Backend Contract

To add LIBERO, implement:

```python
class LiberoBackend(EnvBackend):
    def list_tasks(self) -> list[TaskSpec]: ...
    def reset(self, task_id, instruction=None, seed=None) -> EnvObs: ...
    def step(self, action) -> StepResult: ...
    def render(self) -> dict[str, np.ndarray]: ...
    def close(self) -> None: ...
```

The model side does not need to import LIBERO. It only receives decoded
observations and sends actions through `RemoteEnvClient`.
