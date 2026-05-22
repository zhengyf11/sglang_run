# sglang_run

在标准的 SGLang 镜像容器中运行 SGLang 服务。

本仓库提供一个零第三方 Python 依赖的命令行脚本 `run_sglang_container.py`，用于生成并执行 `docker run` 命令。镜像名、容器名、模型路径/模型 ID、端口映射、GPU、挂载目录、环境变量等常见动态字段都可以通过标准 `--key value` 形式传入。

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
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 30000 \
  --gpus all \
  --dry-run
```

确认命令无误后，去掉 `--dry-run` 启动容器：

```bash
python run_sglang_container.py \
  --image lmsysorg/sglang:latest \
  --container-name sglang-qwen \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 30000 \
  --gpus all
```

默认会生成类似命令：

```bash
docker run --rm --name sglang-qwen --gpus all --shm-size 32g -p 30000:30000 lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000 --tp 1
```

## 常用示例

### 映射本地模型目录

```bash
python run_sglang_container.py \
  --image lmsysorg/sglang:latest \
  --container-name sglang-local-model \
  --model /models/qwen \
  --volume /data/models:/models:ro \
  --host-port 18000 \
  --container-port 30000 \
  --dry-run
```

### 传入环境变量和额外 SGLang 参数

```bash
python run_sglang_container.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --env HF_TOKEN=your_token \
  --env HF_HOME=/root/.cache/huggingface \
  --volume /data/hf-cache:/root/.cache/huggingface \
  --served-model-name qwen2.5 \
  --tp 2 \
  --mem-fraction-static 0.85 \
  --sglang-arg --trust-remote-code
```

### 后台运行并保留退出后的容器

```bash
python run_sglang_container.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --container-name sglang-bg \
  --detach \
  --no-rm
```

### CPU 或不透传 GPU

```bash
python run_sglang_container.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpus none \
  --dry-run
```

## 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | 必填 | 模型路径或模型 ID，传给 SGLang 的 `--model-path`。 |
| `--image` | `lmsysorg/sglang:latest` | Docker 镜像名，可动态指定。 |
| `--container-name` | `sglang` | Docker 容器名，可动态指定。 |
| `--port` | `30000` | 默认服务端口；未指定 `--host-port`/`--container-port` 时同时用于二者。 |
| `--host-port` | 同 `--port` | 宿主机端口。 |
| `--container-port` | 同 `--port` | 容器内 SGLang 端口，同时传给 SGLang `--port`。 |
| `--gpus` | `all` | Docker `--gpus` 值，例如 `all`、`device=0`；设为 `none` 时不添加 `--gpus`。 |
| `--shm-size` | `32g` | Docker `--shm-size` 值。 |
| `--host` | `0.0.0.0` | 容器内 SGLang 监听地址。 |
| `--served-model-name` | 无 | 可选的 SGLang served model name。 |
| `--tp` | `1` | Tensor parallel size，传给 SGLang `--tp`。 |
| `--mem-fraction-static` | 无 | 可选的 SGLang `--mem-fraction-static`。 |
| `--volume` | 可重复 | Docker volume 映射，格式 `HOST:CONTAINER[:MODE]`。 |
| `--env` | 可重复 | Docker 环境变量，格式 `KEY=VALUE`。 |
| `--docker-arg` | 可重复 | 追加到镜像名前的额外 Docker 参数。 |
| `--sglang-arg` | 可重复 | 追加到 SGLang `launch_server` 后的额外参数。 |
| `--detach` | `false` | 使用 `docker run -d` 后台运行。 |
| `--rm` / `--no-rm` | `--rm` | 容器退出后是否自动删除。 |
| `--dry-run` | `false` | 只打印命令，不执行 Docker。 |

## 开发与测试

本项目仅使用 Python 标准库测试框架：

```bash
python -m unittest discover -s tests
```

语法检查：

```bash
python -m py_compile run_sglang_container.py tests/test_run_sglang_container.py
```

## 安全说明

脚本使用 list 形式调用 `subprocess.run(cmd, check=False)`，不使用 `shell=True`。打印命令时只用于人工查看，实际执行不会通过 shell 拼接字符串。
