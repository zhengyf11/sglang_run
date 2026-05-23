# sglang_run

在标准的 SGLang 镜像中启动一个长期运行的 Docker 容器，并提供 p-d 分离 prefill 进程启动脚本。

本仓库提供零第三方 Python 依赖的命令行脚本：

- `run_sglang_container.py`：生成并执行 `docker run` 命令，默认添加 `--privileged` 给容器超级权限，并通过 `--entrypoint /bin/bash` 与 `tail -f /dev/null` 让容器长期运行，不启动任何 SGLang 进程。镜像名、容器名、GPU、共享内存、挂载目录、环境变量等容器层动态字段都可以通过标准 `--key value` 形式传入；网络固定使用 `--net=host`，不提供端口映射参数。
- `run_sglang_prefill.py`：按 Issue #2 的 p-d 分离 prefill 配置启动 `python3 -m sglang.launch_server`，支持通过标准 `--key value` 形式覆盖模型、端口、并行度、disaggregation 与 NCCL 网络参数。

## 环境要求

- Python 3.9+
- Docker
- 如需 GPU：NVIDIA Container Toolkit，并确保 Docker 支持 `--gpus`

## 快速开始

先使用 `--dry-run` 预览将要执行的 Docker 命令：

```bash
python run_sglang_container.py \
  --image lmsysorg/sglang:latest \
  --container-name sglang-qwen \
  --gpus all \
  --dry-run
```

确认命令无误后，去掉 `--dry-run` 启动容器：

```bash
python run_sglang_container.py \
  --image lmsysorg/sglang:latest \
  --container-name sglang-qwen \
  --gpus all
```

默认会生成后台运行并带超级权限的类似命令（脚本始终添加 `-d` 和 `--privileged`，不需要额外参数）：

```bash
docker run --rm -d --name sglang-qwen --gpus all --shm-size 32g --privileged --net=host --entrypoint /bin/bash lmsysorg/sglang:latest -lc 'tail -f /dev/null'
```

## 常用示例

### 映射本地模型目录

```bash
python run_sglang_container.py \
  --image lmsysorg/sglang:latest \
  --container-name sglang-local-model \
  --volume /data/models:/models:ro \
  --dry-run
```

### 传入环境变量和额外 Docker 参数

```bash
python run_sglang_container.py \
  --env HF_TOKEN=your_token \
  --env HF_HOME=/root/.cache/huggingface \
  --volume /data/hf-cache:/root/.cache/huggingface \
  --docker-arg --ipc=host \
  --dry-run
```

### 保留退出后的后台容器

容器默认使用 `docker run -d` 后台运行；如需保留退出后的容器，可搭配 `--no-rm`：

```bash
python run_sglang_container.py \
  --container-name sglang-bg \
  --no-rm
```

### CPU 或不透传 GPU

```bash
python run_sglang_container.py \
  --gpus none \
  --dry-run
```

## 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--image` | `lmsysorg/sglang:latest` | Docker 镜像名，可动态指定。 |
| `--container-name` | `sglang` | Docker 容器名，可动态指定。 |
| `--gpus` | `all` | Docker `--gpus` 值，例如 `all`、`device=0`；设为 `none` 时不添加 `--gpus`。 |
| `--shm-size` | `32g` | Docker `--shm-size` 值。 |
| `--volume` | 可重复 | Docker volume 映射，格式 `HOST:CONTAINER[:MODE]`。 |
| `--env` | 可重复 | Docker 环境变量，格式 `KEY=VALUE`。 |
| `--docker-arg` | 可重复 | 追加到镜像名前的额外 Docker 参数。 |
| 固定 Docker 参数 | `--privileged --net=host --entrypoint /bin/bash` | 脚本默认给容器超级权限，固定使用 host 网络，并用 bash 执行保活命令。 |
| `--rm` / `--no-rm` | `--rm` | 容器退出后是否自动删除；容器始终使用 `docker run -d` 后台运行。 |
| `--dry-run` | `false` | 只打印命令，不执行 Docker。 |

## p-d 分离 prefill 进程启动

`run_sglang_prefill.py` 用于启动 SGLang disaggregation prefill 进程。脚本会在实际执行前清理代理环境变量，设置 NCCL 网络环境变量，并在 Unix/Linux 上尝试执行等价于 `ulimit -l unlimited` 的 `RLIMIT_MEMLOCK` 设置。

### dry-run 预览

建议先用 `--dry-run` 查看将要执行的命令和环境变量，不会启动服务，也不会设置 ulimit：

```bash
python run_sglang_prefill.py --dry-run
```

默认命令使用 Issue #2 中的参数，包括：

- `--model-path /mnt/GLM-5.1-FP8`
- `--served-model-name GLM-5.1-FP8`
- `--tensor-parallel-size 8`
- `--host 0.0.0.0 --port 30000`
- `--disaggregation-mode prefill`
- `--disaggregation-transfer-backend mooncake`
- `--disaggregation-ib-device mlx5_bond_0,...,mlx5_bond_7`
- `--trust-remote-code --disable-cuda-graph`
- `--max-running-requests 128 --chunked-prefill-size 8192 --max-prefill-tokens 65536`

### 覆盖动态参数

常见动态字段都支持标准 `--key value` 形式：

```bash
python run_sglang_prefill.py \
  --model-path /mnt/Custom-Model \
  --served-model-name custom-model \
  --tp 4 \
  --host 0.0.0.0 \
  --port 31000 \
  --nnodes 1 \
  --node-rank 0 \
  --dist-init-addr 127.0.0.1:21000 \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1 \
  --mem-fraction-static 0.75 \
  --max-running-requests 64 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 32768 \
  --dry-run
```

`--tensor-parallel-size` 也可写作 `--tp`。

### NCCL 网络参数

默认 NCCL 环境变量来自 Issue #2，可通过以下参数覆盖：

| 参数 | 默认值 | 对应环境变量 |
| --- | --- | --- |
| `--nccl-ib-gid-index` | `3` | `NCCL_IB_GID_INDEX` |
| `--nccl-ib-hca` | `^mlx5_bond` | `NCCL_IB_HCA` |
| `--nccl-socket-ifname` | `bond0` | `NCCL_SOCKET_IFNAME` |
| `--nccl-ib-tc` | `128` | `NCCL_IB_TC` |
| `--nccl-ib-timeout` | `22` | `NCCL_IB_TIMEOUT` |
| `--nccl-ib-retry-cnt` | `15` | `NCCL_IB_RETRY_CNT` |

脚本会从子进程环境中移除 `http_proxy`、`https_proxy`、`ftp_proxy`、`all_proxy` 及其大写形式。

### 其他 prefill 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--tool-call-parser` | `glm47` | 传给 SGLang 的 tool call parser。 |
| `--reasoning-parser` | `glm45` | 传给 SGLang 的 reasoning parser。 |
| `--speculative-algorithm` | `EAGLE` | Speculative decoding 算法。 |
| `--speculative-num-steps` | `3` | Speculative steps。 |
| `--speculative-eagle-topk` | `1` | EAGLE top-k。 |
| `--speculative-num-draft-tokens` | `4` | Draft token 数。 |
| `--trust-remote-code` / `--no-trust-remote-code` | 启用 | 是否传递 `--trust-remote-code`。 |
| `--disable-cuda-graph` / `--enable-cuda-graph` | 禁用 CUDA graph | 是否传递 `--disable-cuda-graph`。 |
| `--sglang-arg` | 可重复 | 追加额外 `launch_server` 参数；支持 `--sglang-arg --log-level --sglang-arg debug`。 |
| `--skip-ulimit` | `false` | 跳过 `RLIMIT_MEMLOCK` 设置；适用于已由外部环境配置或非 Unix 平台。 |

实际执行使用 list 形式 `subprocess.run(cmd, env=env, check=False)`，不使用 `shell=True`。

## 开发与测试

本项目仅使用 Python 标准库测试框架：

```bash
python -m unittest discover -s tests
```

语法检查：

```bash
python -m py_compile run_sglang_container.py run_sglang_prefill.py tests/test_run_sglang_container.py tests/test_run_sglang_prefill.py
```

prefill dry-run 验证：

```bash
python run_sglang_prefill.py --dry-run
```

## 安全说明

脚本使用 list 形式调用 `subprocess.run(...)`，不使用 `shell=True`。打印命令时只用于人工查看，实际执行不会通过 shell 拼接字符串。
