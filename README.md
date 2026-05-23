# sglang_run

在标准的 SGLang 镜像中启动一个长期运行的 Docker 容器，并提供 p-d 分离 prefill 命令生成 Web UI。

本仓库提供零第三方 Python 依赖的脚本：

- `run_sglang_container.py`：生成并执行 `docker run` 命令，默认添加 root 用户、特权模式、host IPC/网络、NVIDIA runtime/GPU、memlock ulimit、cgroup 只读挂载与 `NVIDIA_VISIBLE_DEVICES=all`，并通过 `--entrypoint /bin/bash` 与 `tail -f /dev/null` 让容器长期运行，不启动任何 SGLang 进程。镜像名、容器名、共享内存、额外挂载目录、额外环境变量等容器层动态字段都可以通过标准 `--key value` 形式传入；网络固定使用 `--network host`，不提供端口映射参数。
- `prefill_command_web.py`：启动一个本地 Web 页面，按 Issue #2 的 p-d 分离 prefill 默认配置实时生成 `python3 -m sglang.launch_server` shell 命令。它只生成命令文本，不直接拉起 prefill，不调用 `subprocess.run`，也不提供真实执行接口。

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
  --dry-run
```

确认命令无误后，去掉 `--dry-run` 启动容器：

```bash
python run_sglang_container.py \
  --image lmsysorg/sglang:latest \
  --container-name sglang-qwen
```

默认会生成后台运行并带固定容器运行参数的类似命令（脚本始终添加 `-d`、root 用户、特权模式、host IPC/网络、NVIDIA runtime/GPU、memlock ulimit、cgroup 只读挂载和 `NVIDIA_VISIBLE_DEVICES=all`，不需要额外参数）：

```bash
docker run --rm -d --name sglang-qwen --shm-size 32g --user=0 --privileged --ipc=host --network host --runtime=nvidia --gpus all --ulimit memlock=-1:-1 -v /sys/fs/cgroup:/sys/fs/cgroup:ro -e NVIDIA_VISIBLE_DEVICES=all --entrypoint /bin/bash lmsysorg/sglang:latest -lc 'tail -f /dev/null'
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

### 传入额外环境变量、挂载和 Docker 参数

```bash
python run_sglang_container.py \
  --env HF_TOKEN=your_token \
  --env HF_HOME=/root/.cache/huggingface \
  --volume /data/hf-cache:/root/.cache/huggingface \
  --docker-arg --log-level=debug \
  --dry-run
```

### 保留退出后的后台容器

容器默认使用 `docker run -d` 后台运行；如需保留退出后的容器，可搭配 `--no-rm`：

```bash
python run_sglang_container.py \
  --container-name sglang-bg \
  --no-rm
```


## 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--image` | `lmsysorg/sglang:latest` | Docker 镜像名，可动态指定。 |
| `--container-name` | `sglang` | Docker 容器名，可动态指定。 |
| `--shm-size` | `32g` | Docker `--shm-size` 值。 |
| `--volume` | 可重复 | Docker volume 映射，格式 `HOST:CONTAINER[:MODE]`。 |
| `--env` | 可重复 | Docker 环境变量，格式 `KEY=VALUE`。 |
| `--docker-arg` | 可重复 | 追加到镜像名前的额外 Docker 参数。 |
| 固定 Docker 参数 | `--user=0 --privileged --ipc=host --network host --runtime=nvidia --gpus all --ulimit memlock=-1:-1 -v /sys/fs/cgroup:/sys/fs/cgroup:ro -e NVIDIA_VISIBLE_DEVICES=all --entrypoint /bin/bash` | 脚本默认使用 root 用户、特权模式、host IPC/网络、NVIDIA runtime/GPU、memlock ulimit、cgroup 只读挂载，并用 bash 执行保活命令。 |
| `--rm` / `--no-rm` | `--rm` | 容器退出后是否自动删除；容器始终使用 `docker run -d` 后台运行。 |
| `--dry-run` | `false` | 只打印命令，不执行 Docker。 |

## p-d 分离 prefill 命令生成 Web UI

`prefill_command_web.py` 会启动一个本地网站，用于生成 SGLang disaggregation prefill 启动命令。用户可以在页面中填写或选择参数；字段为空、缺失或 `null` 时使用 Issue #2 默认值。页面会调用本地 API 实时刷新输出窗口中的 shell 命令。

Web UI 默认监听浏览器安全端口 `6060`，也可以通过 `--port` 显式覆盖：

```bash
python prefill_command_web.py --host 127.0.0.1
# 或者指定其他端口
python prefill_command_web.py --host 127.0.0.1 --port 8080
```

打开：

```text
http://127.0.0.1:6060/
```

页面代码位于 `web/` 目录（`index.html`、`styles.css`、`app.js`），Python 只负责提供静态文件和本地 API。页面包含参数分组表单、实时命令输出、复制按钮、状态提示和 API 错误提示。页面和 API 只生成命令文本，不会启动 SGLang，不会设置 ulimit，不会清理当前进程环境，也不会调用 `subprocess.run`。

### 默认生成的 prefill 参数

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

页面也会生成 NCCL export 和代理环境变量 unset 提示，供人工复制使用：

| 环境变量 | 默认值 |
| --- | --- |
| `NCCL_IB_GID_INDEX` | `3` |
| `NCCL_IB_HCA` | `^mlx5_bond` |
| `NCCL_SOCKET_IFNAME` | `bond0` |
| `NCCL_IB_TC` | `128` |
| `NCCL_IB_TIMEOUT` | `22` |
| `NCCL_IB_RETRY_CNT` | `15` |

### 本地 API

Web UI 使用以下本地接口：

- `GET /`：返回包含参数表单和输出窗口的 HTML 页面。
- `GET /api/defaults`：返回 Issue #2 默认配置。
- `POST /api/command`：接收 JSON 对象，返回 `command` list、`shell_command`、`env_exports`、`proxy_unsets` 与 `combined_shell`。

`POST /api/command` 示例：

```bash
curl -s http://127.0.0.1:6060/api/command \
  -H 'Content-Type: application/json' \
  -d '{"model_path":"/mnt/Custom-Model","served_model_name":"custom-model","tensor_parallel_size":4,"extra_sglang_args":"--log-level debug"}'
```

额外 SGLang 参数在页面的 `Extra SGLang args` 文本框填写，后端使用 `shlex.split()` 解析并追加到命令末尾；无法解析时 API 返回 400，不生成错误命令。

## 开发与测试

本项目仅使用 Python 标准库测试框架：

```bash
python -m unittest discover -s tests -v
```

语法检查：

```bash
python -m py_compile run_sglang_container.py prefill_command_web.py tests/test_run_sglang_container.py tests/test_prefill_command_web.py
```

## 安全说明

- `run_sglang_container.py` 使用 list 形式调用 `subprocess.run(...)`，不使用 `shell=True`。
- `prefill_command_web.py` 不导入 `subprocess`，不执行 `sglang.launch_server`，不实现 `/start`、`/run` 或 `execute=true` 等真实执行入口；生成的 shell command 仅供人工复制和审阅。
