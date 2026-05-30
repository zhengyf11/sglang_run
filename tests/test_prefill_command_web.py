from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from unittest import mock

import prefill_command_web


def option_values(command: list[str], option: str) -> list[str]:
    return [command[index + 1] for index, item in enumerate(command[:-1]) if item == option]


def assert_option_block(testcase: unittest.TestCase, command: list[str], option: str, expected_values: list[str]) -> None:
    option_indexes = [index for index, item in enumerate(command) if item == option]
    testcase.assertEqual(option_indexes, list(range(option_indexes[0], option_indexes[0] + len(option_indexes) * 2, 2)))
    testcase.assertEqual([command[index + 1] for index in option_indexes], expected_values)


class CommandGenerationTests(unittest.TestCase):
    def test_builds_default_issue_command(self) -> None:
        response = prefill_command_web.build_command_response({})
        cmd = response["command"]

        self.assertEqual(cmd[:3], ["python3", "-m", "sglang.launch_server"])
        self.assertEqual(cmd[cmd.index("--model-path") + 1], "/mnt/GLM-5.1-FP8")
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "GLM-5.1-FP8")
        self.assertEqual(cmd[cmd.index("--tp-size") + 1], "8")
        self.assertEqual(cmd.count("--tp-size"), 1)
        self.assertNotIn("--tensor-parallel-size", cmd)
        self.assertEqual(cmd[cmd.index("--tool-call-parser") + 1], "glm47")
        self.assertEqual(cmd[cmd.index("--reasoning-parser") + 1], "glm45")
        self.assertNotIn("--speculative-algorithm", cmd)
        self.assertNotIn("--speculative-num-steps", cmd)
        self.assertNotIn("--speculative-eagle-topk", cmd)
        self.assertNotIn("--speculative-num-draft-tokens", cmd)
        self.assertEqual(cmd[cmd.index("--mem-fraction-static") + 1], "0.9")
        self.assertEqual(cmd[cmd.index("--host") + 1], "0.0.0.0")
        self.assertEqual(cmd[cmd.index("--port") + 1], "30000")
        self.assertEqual(cmd[cmd.index("--nnodes") + 1], "1")
        self.assertEqual(cmd[cmd.index("--node-rank") + 1], "0")
        self.assertEqual(cmd[cmd.index("--dist-init-addr") + 1], "127.0.0.1:20000")
        self.assertEqual(cmd[cmd.index("--disaggregation-mode") + 1], "prefill")
        self.assertEqual(cmd[cmd.index("--disaggregation-transfer-backend") + 1], "mooncake")
        self.assertEqual(
            cmd[cmd.index("--disaggregation-ib-device") + 1],
            "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,"
            "mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7",
        )
        self.assertIn("--trust-remote-code", cmd)
        self.assertIn("--disable-cuda-graph", cmd)
        self.assertNotIn("--max-running-requests", cmd)
        self.assertNotIn("--cuda-graph-max-bs", cmd)
        self.assertEqual(cmd[cmd.index("--chunked-prefill-size") + 1], "8192")
        self.assertEqual(cmd[cmd.index("--max-prefill-tokens") + 1], "65536")
        self.assertFalse(response["executed"])
        self.assertIn("python3 -m sglang.launch_server", response["shell_command"])

    def test_shell_command_formats_launch_args_on_separate_lines(self) -> None:
        response = prefill_command_web.build_command_response(
            {
                "model_path": "/mnt/GLM-5.1-FP8",
                "served_model_name": "GLM-5.1-FP8",
                "parallel_tp_size": 8,
            }
        )

        shell_lines = response["shell_command"].splitlines()

        self.assertEqual(shell_lines[0], "python3 -m sglang.launch_server \\")
        self.assertEqual(shell_lines[1], "\t--model-path /mnt/GLM-5.1-FP8 \\")
        self.assertEqual(shell_lines[2], "\t--served-model-name GLM-5.1-FP8 \\")
        self.assertEqual(shell_lines[3], "\t--tp-size 8 \\")
        self.assertTrue(all(line.endswith(" \\") for line in shell_lines[:-1]))
        self.assertFalse(shell_lines[-1].endswith("\\"))
        self.assertEqual(
            response["command"][:9],
            [
                "python3",
                "-m",
                "sglang.launch_server",
                "--model-path",
                "/mnt/GLM-5.1-FP8",
                "--served-model-name",
                "GLM-5.1-FP8",
                "--tp-size",
                "8",
            ],
        )

    def test_blank_missing_and_none_fields_use_defaults(self) -> None:
        config = prefill_command_web.normalize_form_payload(
            {"model_path": "", "served_model_name": None, "parallel_tp_size": "  "}
        )

        self.assertEqual(config["model_path"], "/mnt/GLM-5.1-FP8")
        self.assertEqual(config["served_model_name"], "GLM-5.1-FP8")
        self.assertEqual(config["tool_call_parser"], "glm47")
        self.assertEqual(config["reasoning_parser"], "glm45")
        self.assertEqual(config["parallel_tp_size"], 8)
        self.assertEqual(config["dp_size"], 8)
        self.assertEqual(config["attention_parallel_mode"], "tensor")
        self.assertEqual(config["context_parallel_backend"], "nsa")
        self.assertTrue(config["enable_dynamic_chunking"])
        self.assertEqual(config["dynamic_chunking_smooth_factor"], "0.8")
        self.assertEqual(config["moe_parallel_mode"], "tensor")
        self.assertFalse(config["enable_single_batch_overlap"])
        self.assertFalse(config["enable_two_batch_overlap"])
        self.assertEqual(config["port"], 30000)

    def test_dynamic_parameters_override_defaults(self) -> None:
        response = prefill_command_web.build_command_response(
            {
                "model_path": "/models/custom",
                "served_model_name": "custom-model",
                "parallel_tp_size": 2,
                "host": "127.0.0.1",
                "port": 31000,
                "nnodes": 2,
                "node_rank": 1,
                "dist_init_addr": "10.0.0.1:20000",
                "disaggregation_transfer_backend": "custom-backend",
                "disaggregation_ib_device": "mlx5_0",
                "mem_fraction_static": 0.55,
                "chunked_prefill_size": 4096,
                "max_prefill_tokens": 32768,
            }
        )
        cmd = response["command"]

        self.assertEqual(cmd[cmd.index("--model-path") + 1], "/models/custom")
        self.assertEqual(cmd[cmd.index("--served-model-name") + 1], "custom-model")
        self.assertEqual(cmd[cmd.index("--tp-size") + 1], "2")
        self.assertNotIn("--tensor-parallel-size", cmd)
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")
        self.assertEqual(cmd[cmd.index("--port") + 1], "31000")
        self.assertEqual(cmd[cmd.index("--nnodes") + 1], "1")
        self.assertEqual(cmd[cmd.index("--node-rank") + 1], "0")
        self.assertEqual(cmd[cmd.index("--dist-init-addr") + 1], "10.0.0.1:20000")
        self.assertEqual(cmd[cmd.index("--disaggregation-transfer-backend") + 1], "custom-backend")
        self.assertEqual(cmd[cmd.index("--disaggregation-ib-device") + 1], "mlx5_0")
        self.assertEqual(cmd[cmd.index("--mem-fraction-static") + 1], "0.55")
        self.assertNotIn("--max-running-requests", cmd)
        self.assertNotIn("--cuda-graph-max-bs", cmd)
        self.assertEqual(cmd[cmd.index("--chunked-prefill-size") + 1], "4096")
        self.assertEqual(cmd[cmd.index("--max-prefill-tokens") + 1], "32768")

    def test_disaggregation_mode_is_fixed_to_prefill(self) -> None:
        response = prefill_command_web.build_command_response({"disaggregation_mode": "decode"})
        cmd = response["command"]

        self.assertEqual(response["config"]["disaggregation_mode"], "prefill")
        self.assertEqual(cmd[cmd.index("--disaggregation-mode") + 1], "prefill")

    def test_served_model_name_is_inferred_from_model_path(self) -> None:
        cases = (
            ("/mnt/GLM-5.1-FP8", "GLM-5.1-FP8"),
            ("/mnt/models/Qwen3-235B-A22B/v1", "Qwen3-235B-A22B"),
            ("/models/Qwen3-Coder-480B-A35B-Instruct", "Qwen3-Coder-480B-A35B-Instruct"),
            (r"D:\models\DeepSeek-V3.2\v1", "DeepSeek-V3.2"),
            ("/models/unknown-model", "unknown-model"),
        )
        for model_path, expected_name in cases:
            with self.subTest(model_path=model_path):
                config = prefill_command_web.normalize_form_payload({"model_path": model_path})

                self.assertEqual(config["served_model_name"], expected_name)

    def test_model_path_infers_registered_parser_defaults(self) -> None:
        cases = (
            ("/mnt/GLM-5.1-FP8", "glm47", "glm45"),
            ("/mnt/models/Qwen3-235B-A22B/v1", "qwen", "qwen3"),
            ("/models/Qwen3-Coder-480B-A35B-Instruct", "qwen3_coder", "qwen3"),
            (r"D:\models\DeepSeek-V3.2\v1", "deepseekv32", "deepseek-v3"),
            ("/models/MiniMax-M2.7", "minimax-m2", "minimax"),
            ("MiniMaxAI/MiniMax-M2.7", "minimax-m2", "minimax"),
            (r"D:\models\MiniMax-M2-BF16\v1", "minimax-m2", "minimax"),
            ("/models/unknown-model", "unknown", "unknown"),
        )
        tool_choices = prefill_command_web.get_tool_call_parser_choices()
        reasoning_choices = prefill_command_web.get_reasoning_parser_choices()
        for model_path, expected_tool, expected_reasoning in cases:
            with self.subTest(model_path=model_path):
                config = prefill_command_web.normalize_form_payload({"model_path": model_path})

                self.assertEqual(config["tool_call_parser"], expected_tool)
                self.assertEqual(config["reasoning_parser"], expected_reasoning)
                self.assertIn(config["tool_call_parser"], tool_choices)
                self.assertIn(config["reasoning_parser"], reasoning_choices)

    def test_invalid_parser_payload_values_fall_back_to_inferred_registered_values(self) -> None:
        config = prefill_command_web.normalize_form_payload(
            {
                "model_path": "/mnt/models/Qwen3-235B-A22B/v1",
                "tool_call_parser": "not-a-parser",
                "reasoning_parser": "not-a-parser",
            }
        )

        self.assertEqual(config["tool_call_parser"], "qwen")
        self.assertEqual(config["reasoning_parser"], "qwen3")

    def test_unknown_parser_payload_values_are_preserved(self) -> None:
        config = prefill_command_web.normalize_form_payload(
            {
                "model_path": "/mnt/models/Qwen3-235B-A22B/v1",
                "tool_call_parser": "unknown",
                "reasoning_parser": "unknown",
            }
        )

        self.assertEqual(config["tool_call_parser"], "unknown")
        self.assertEqual(config["reasoning_parser"], "unknown")

    def test_attention_dp_parallel_defaults_dp_size_to_world_size(self) -> None:
        cmd = prefill_command_web.build_command_response(
            {"parallel_tp_size": 8, "attention_parallel_mode": "dp_attention", "dp_size": ""}
        )["command"]

        self.assertEqual(cmd[cmd.index("--tp-size") + 1], "8")
        self.assertEqual(cmd[cmd.index("--dp-size") + 1], "8")
        self.assertIn("--enable-dp-attention", cmd)

    def test_attention_dp_parallel_uses_custom_dp_size(self) -> None:
        cmd = prefill_command_web.build_command_response(
            {"parallel_tp_size": 8, "attention_parallel_mode": "dp_attention", "dp_size": 2}
        )["command"]

        self.assertEqual(cmd[cmd.index("--tp-size") + 1], "8")
        self.assertEqual(cmd[cmd.index("--dp-size") + 1], "2")
        self.assertIn("--enable-dp-attention", cmd)

    def test_context_parallel_selects_backend_and_deduplicates_overlap(self) -> None:
        cmd = prefill_command_web.build_command_response(
            {
                "parallel_tp_size": 8,
                "attention_parallel_mode": "context_parallel",
                "context_parallel_backend": "prefill",
                "moe_parallel_mode": "expert_parallel",
                "enable_single_batch_overlap": True,
                "enable_two_batch_overlap": True,
            }
        )["command"]

        self.assertEqual(cmd[cmd.index("--tp-size") + 1], "8")
        self.assertIn("--enable-prefill-context-parallel", cmd)
        self.assertNotIn("--enable-nsa-prefill-context-parallel", cmd)
        self.assertEqual(cmd[cmd.index("--nsa-prefill-cp-mode") + 1], "in-seq-split")
        self.assertEqual(cmd.count("--tp-size"), 1)
        self.assertEqual(cmd.count("--enable-two-batch-overlap"), 1)
        self.assertIn("--ep-size=8", cmd)
        self.assertEqual(cmd[cmd.index("--moe-a2a-backend") + 1], "deepep")
        self.assertIn("--enable-single-batch-overlap", cmd)

    def test_moe_tensor_parallel_does_not_add_overlap_flags(self) -> None:
        cmd = prefill_command_web.build_command_response(
            {"parallel_tp_size": 8, "moe_parallel_mode": "tensor"}
        )["command"]

        self.assertEqual(cmd.count("--tp-size"), 1)
        self.assertNotIn("--ep-size=8", cmd)
        self.assertNotIn("--moe-a2a-backend", cmd)
        self.assertNotIn("--enable-single-batch-overlap", cmd)
        self.assertNotIn("--enable-two-batch-overlap", cmd)

    def test_moe_expert_parallel_omits_overlap_flags_by_default(self) -> None:
        cmd = prefill_command_web.build_command_response(
            {"parallel_tp_size": 8, "moe_parallel_mode": "expert_parallel"}
        )["command"]

        self.assertEqual(cmd.count("--tp-size"), 1)
        self.assertIn("--ep-size=8", cmd)
        self.assertEqual(cmd[cmd.index("--moe-a2a-backend") + 1], "deepep")
        self.assertNotIn("--enable-single-batch-overlap", cmd)
        self.assertNotIn("--enable-two-batch-overlap", cmd)

    def test_moe_expert_parallel_adds_selected_overlap_flags(self) -> None:
        cmd = prefill_command_web.build_command_response(
            {
                "parallel_tp_size": 8,
                "moe_parallel_mode": "expert_parallel",
                "enable_single_batch_overlap": True,
                "enable_two_batch_overlap": True,
            }
        )["command"]

        self.assertEqual(cmd.count("--tp-size"), 1)
        self.assertIn("--ep-size=8", cmd)
        self.assertEqual(cmd[cmd.index("--moe-a2a-backend") + 1], "deepep")
        self.assertIn("--enable-single-batch-overlap", cmd)
        self.assertEqual(cmd.count("--enable-two-batch-overlap"), 1)

    def test_context_parallel_defaults_to_nsa_backend(self) -> None:
        cmd = prefill_command_web.build_command_response(
            {"attention_parallel_mode": "context_parallel"}
        )["command"]

        self.assertIn("--enable-nsa-prefill-context-parallel", cmd)
        self.assertNotIn("--enable-prefill-context-parallel", cmd)

    def test_pipeline_parallel_adds_dynamic_chunking_shell_hint(self) -> None:
        response = prefill_command_web.build_command_response(
            {"parallel_tp_size": 8, "attention_parallel_mode": "pipeline_parallel"}
        )
        cmd = response["command"]

        self.assertEqual(cmd[cmd.index("--tp-size") + 1], "1")
        self.assertEqual(cmd[cmd.index("--pp-size") + 1], "8")
        self.assertIn("--enable-dynamic-chunking", cmd)
        self.assertIn("#export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.8", response["combined_shell"])

    def test_pipeline_parallel_uses_custom_dynamic_chunking_smooth_factor(self) -> None:
        response = prefill_command_web.build_command_response(
            {
                "parallel_tp_size": 4,
                "attention_parallel_mode": "pipeline_parallel",
                "dynamic_chunking_smooth_factor": "0.6",
            }
        )

        self.assertIn("--enable-dynamic-chunking", response["command"])
        self.assertIn("#export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.6", response["combined_shell"])
        self.assertNotIn("#export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.8", response["combined_shell"])

    def test_pipeline_parallel_can_disable_dynamic_chunking(self) -> None:
        response = prefill_command_web.build_command_response(
            {
                "parallel_tp_size": 8,
                "attention_parallel_mode": "pipeline_parallel",
                "enable_dynamic_chunking": False,
                "dynamic_chunking_smooth_factor": "0.6",
            }
        )
        cmd = response["command"]

        self.assertEqual(cmd[cmd.index("--tp-size") + 1], "1")
        self.assertEqual(cmd[cmd.index("--pp-size") + 1], "8")
        self.assertNotIn("--enable-dynamic-chunking", cmd)
        self.assertNotIn("SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR", response["combined_shell"])

    def test_mtp_parameters_are_omitted_until_enabled(self) -> None:
        response = prefill_command_web.build_command_response(
            {
                "enable_mtp": False,
                "speculative_algorithm": "EAGLE",
                "speculative_num_steps": 5,
                "speculative_eagle_topk": 2,
                "speculative_num_draft_tokens": 8,
            }
        )
        cmd = response["command"]

        self.assertFalse(response["config"]["enable_mtp"])
        self.assertNotIn("--speculative-algorithm", cmd)
        self.assertNotIn("--speculative-num-steps", cmd)
        self.assertNotIn("--speculative-eagle-topk", cmd)
        self.assertNotIn("--speculative-num-draft-tokens", cmd)

    def test_mtp_parameters_are_included_when_enabled(self) -> None:
        response = prefill_command_web.build_command_response(
            {
                "enable_mtp": True,
                "speculative_algorithm": "EAGLE",
                "speculative_num_steps": 5,
                "speculative_eagle_topk": 2,
                "speculative_num_draft_tokens": 8,
            }
        )
        cmd = response["command"]

        self.assertTrue(response["config"]["enable_mtp"])
        self.assertEqual(cmd[cmd.index("--speculative-algorithm") + 1], "EAGLE")
        self.assertEqual(cmd[cmd.index("--speculative-num-steps") + 1], "5")
        self.assertEqual(cmd[cmd.index("--speculative-eagle-topk") + 1], "2")
        self.assertEqual(cmd[cmd.index("--speculative-num-draft-tokens") + 1], "8")

    def test_prefill_trust_remote_code_is_fixed_when_payload_disables_it(self) -> None:
        response = prefill_command_web.build_command_response(
            {"trust_remote_code": False, "disable_cuda_graph": False}
        )
        cmd = response["command"]

        self.assertTrue(response["config"]["trust_remote_code"])
        self.assertIn("--trust-remote-code", cmd)
        self.assertNotIn("--disable-cuda-graph", cmd)

    def test_extra_sglang_args_are_appended(self) -> None:
        cmd = prefill_command_web.build_command_response(
            {"extra_sglang_args": "--log-level debug --foo 'bar baz'"}
        )["command"]

        self.assertEqual(cmd[-4:], ["--log-level", "debug", "--foo", "bar baz"])

    def test_invalid_extra_sglang_args_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            prefill_command_web.build_command_response({"extra_sglang_args": "--unterminated '"})

    def test_env_exports_and_proxy_unsets_are_display_only(self) -> None:
        response = prefill_command_web.build_command_response(
            {
                "NCCL_IB_GID_INDEX": "5",
                "NCCL_IB_HCA": "mlx5_0",
                "NCCL_IB_TC": "64",
                "NCCL_IB_TIMEOUT": "10",
                "NCCL_IB_RETRY_CNT": "3",
            }
        )

        self.assertIn("export NCCL_IB_GID_INDEX=3", response["env_exports"])
        self.assertIn("export NCCL_IB_HCA=mlx5_0", response["env_exports"])
        self.assertIn("export NCCL_IB_TC=128", response["env_exports"])
        self.assertIn("export NCCL_IB_TIMEOUT=22", response["env_exports"])
        self.assertIn("export NCCL_IB_RETRY_CNT=15", response["env_exports"])
        self.assertIn("unset http_proxy", response["proxy_unsets"])
        self.assertIn("unset HTTPS_PROXY", response["proxy_unsets"])

    def test_decode_profile_builds_issue_5_command_and_combined_shell(self) -> None:
        response = prefill_command_web.build_command_response({}, profile="decode")
        cmd = response["command"]

        self.assertEqual(response["profile"], "decode")
        self.assertEqual(cmd[:3], ["python3", "-m", "sglang.launch_server"])
        self.assertEqual(cmd[cmd.index("--port") + 1], "30001")
        self.assertEqual(cmd[cmd.index("--disaggregation-mode") + 1], "decode")
        self.assertEqual(cmd[cmd.index("--mem-fraction-static") + 1], "0.7")
        self.assertEqual(cmd[cmd.index("--max-running-requests") + 1], "128")
        self.assertEqual(cmd[cmd.index("--cuda-graph-max-bs") + 1], "128")
        self.assertIn("--speculative-algorithm", cmd)
        self.assertIn("--trust-remote-code", cmd)
        self.assertNotIn("--disable-cuda-graph", cmd)
        self.assertIn("ulimit -l unlimited", response["combined_shell"])
        self.assertIn("unset http_proxy", response["combined_shell"])
        self.assertIn("export NCCL_IB_GID_INDEX=3", response["combined_shell"])

    def test_decode_profile_trust_remote_code_is_fixed_when_payload_disables_it(self) -> None:
        response = prefill_command_web.build_command_response({"trust_remote_code": False}, profile="decode")
        cmd = response["command"]

        self.assertTrue(response["config"]["trust_remote_code"])
        self.assertIn("--trust-remote-code", cmd)

    def test_decode_profile_disable_cuda_graph_is_opt_in(self) -> None:
        default_response = prefill_command_web.build_command_response({}, profile="decode")
        enabled_response = prefill_command_web.build_command_response({"disable_cuda_graph": True}, profile="decode")

        self.assertFalse(default_response["config"]["disable_cuda_graph"])
        self.assertNotIn("--disable-cuda-graph", default_response["command"])
        self.assertTrue(enabled_response["config"]["disable_cuda_graph"])
        self.assertIn("--disable-cuda-graph", enabled_response["command"])

    def test_decode_profile_allows_limit_overrides(self) -> None:
        response = prefill_command_web.build_command_response(
            {"max_running_requests": 64, "cuda_graph_max_bs": 256},
            profile="decode",
        )
        cmd = response["command"]

        self.assertEqual(response["config"]["max_running_requests"], 64)
        self.assertEqual(response["config"]["cuda_graph_max_bs"], 256)
        self.assertEqual(cmd[cmd.index("--max-running-requests") + 1], "64")
        self.assertEqual(cmd[cmd.index("--cuda-graph-max-bs") + 1], "256")

    def test_decode_disaggregation_mode_is_fixed(self) -> None:
        response = prefill_command_web.build_command_response({"disaggregation_mode": "prefill"}, profile="decode")

        self.assertEqual(response["config"]["disaggregation_mode"], "decode")
        self.assertEqual(response["command"][response["command"].index("--disaggregation-mode") + 1], "decode")

    def test_router_profile_builds_issue_8_command_without_nccl_exports(self) -> None:
        response = prefill_command_web.build_command_response({}, profile="router")
        cmd = response["command"]

        self.assertEqual(response["profile"], "router")
        self.assertEqual(cmd[:4], ["python3", "-m", "sglang_router.launch_router", "--pd-disaggregation"])
        self.assertEqual(cmd[cmd.index("--prefill") + 1], "http://192.168.1.99:30000")
        self.assertEqual(cmd[cmd.index("--decode") + 1], "http://192.168.1.233:30001")
        self.assertEqual(cmd[cmd.index("--port") + 1], "8000")
        self.assertEqual(cmd[cmd.index("--policy") + 1], "cache_aware")
        self.assertEqual(cmd[cmd.index("--tool-call-parser") + 1], "glm47_moe")
        self.assertEqual(cmd[cmd.index("--reasoning-parser") + 1], "glm45")
        self.assertEqual(cmd[cmd.index("--retry-max-retries") + 1], "3")
        self.assertIn("unset http_proxy", response["combined_shell"])
        self.assertNotIn("NCCL_IB_GID_INDEX", response["combined_shell"])
        self.assertNotIn("--trust-remote-code", cmd)
        self.assertNotIn("--disable-cuda-graph", cmd)
        self.assertNotIn("--max-running-requests", cmd)
        self.assertNotIn("--cuda-graph-max-bs", cmd)

    def test_router_profile_allows_endpoint_overrides(self) -> None:
        response = prefill_command_web.build_command_response(
            {"prefill": "http://127.0.0.1:30000", "decode": "http://127.0.0.1:30001"},
            profile="router",
        )
        cmd = response["command"]

        self.assertEqual(cmd[cmd.index("--prefill") + 1], "http://127.0.0.1:30000")
        self.assertEqual(cmd[cmd.index("--decode") + 1], "http://127.0.0.1:30001")

    def test_docker_run_profile_builds_default_keepalive_command(self) -> None:
        response = prefill_command_web.build_command_response({}, profile="docker_run")
        cmd = response["command"]

        self.assertEqual(response["profile"], "docker_run")
        self.assertEqual(cmd[:4], ["docker", "run", "--rm", "-d"])
        self.assertIn("--user=0", cmd)
        self.assertIn("--privileged", cmd)
        self.assertIn("--ipc=host", cmd)
        self.assertEqual(cmd[cmd.index("--network") + 1], "host")
        self.assertIn("--runtime=nvidia", cmd)
        self.assertEqual(cmd[cmd.index("--gpus") + 1], "all")
        self.assertEqual(cmd[cmd.index("--ulimit") + 1], "memlock=-1:-1")
        self.assertIn("/sys/fs/cgroup:/sys/fs/cgroup:ro", cmd)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=all", cmd)
        self.assertIn("/mnt/GLM-5.1-FP8:/mnt/GLM-5.1-FP8", cmd)
        assert_option_block(
            self,
            cmd,
            "-v",
            ["/sys/fs/cgroup:/sys/fs/cgroup:ro", "/mnt/GLM-5.1-FP8:/mnt/GLM-5.1-FP8"],
        )
        assert_option_block(self, cmd, "-e", ["NVIDIA_VISIBLE_DEVICES=all"])
        self.assertLess(cmd.index("/mnt/GLM-5.1-FP8:/mnt/GLM-5.1-FP8"), cmd.index("-e"))
        self.assertLess(cmd.index("NVIDIA_VISIBLE_DEVICES=all"), cmd.index("--entrypoint"))
        self.assertEqual(cmd[cmd.index("--entrypoint") + 1], "/bin/bash")
        self.assertEqual(cmd[-3:], ["lmsysorg/sglang:latest", "-lc", "tail -f /dev/null"])
        self.assertIn("-v /sys/fs/cgroup:/sys/fs/cgroup:ro \\", response["shell_command"])
        self.assertIn("-v /mnt/GLM-5.1-FP8:/mnt/GLM-5.1-FP8 \\", response["shell_command"])
        self.assertIn("-e NVIDIA_VISIBLE_DEVICES=all \\", response["shell_command"])
        self.assertNotIn("-v \\\n", response["shell_command"])
        self.assertNotIn("-e \\\n", response["shell_command"])
        self.assertFalse(response["executed"])
        self.assertEqual(response["resource_limits"], [])
        self.assertEqual(response["env_exports"], [])

    def test_docker_run_profile_supports_dynamic_fields_and_ignores_web_docker_arg(self) -> None:
        response = prefill_command_web.build_command_response(
            {
                "image": "custom/sglang:test",
                "container_name": "custom-name",
                "shm_size": "16g",
                "rm": False,
                "volume": "/data/models:/models:ro\n/data/cache:/cache\n",
                "env": "HF_HOME=/cache\nTOKEN=value with spaces",
                "docker_arg": "--cap-add=SYS_ADMIN\n--security-opt=seccomp=unconfined",
                "model_path": "/models/custom",
            },
            profile="docker_run",
        )
        cmd = response["command"]

        self.assertNotIn("--rm", cmd)
        self.assertEqual(cmd[cmd.index("--name") + 1], "custom-name")
        self.assertEqual(cmd[cmd.index("--shm-size") + 1], "16g")
        self.assertIn("/data/models:/models:ro", cmd)
        self.assertIn("/data/cache:/cache", cmd)
        self.assertNotIn("/models/custom:/models/custom", cmd)
        self.assertIn("HF_HOME=/cache", cmd)
        self.assertIn("TOKEN=value with spaces", cmd)
        assert_option_block(self, cmd, "-v", ["/data/models:/models:ro", "/data/cache:/cache"])
        assert_option_block(self, cmd, "-e", ["HF_HOME=/cache", "TOKEN=value with spaces"])
        self.assertNotIn("--cap-add=SYS_ADMIN", cmd)
        self.assertNotIn("--security-opt=seccomp=unconfined", cmd)
        self.assertNotIn("docker_arg", response["config"])
        self.assertEqual(cmd[-3:], ["custom/sglang:test", "-lc", "tail -f /dev/null"])


    def test_docker_run_explicit_empty_volume_and_env_are_preserved(self) -> None:
        response = prefill_command_web.build_command_response(
            {"volume": [], "env": [], "model_path": "/models/custom"},
            profile="docker_run",
        )
        cmd = response["command"]

        self.assertEqual(response["config"]["volume"], [])
        self.assertEqual(response["config"]["env"], [])
        self.assertNotIn("-v", cmd)
        self.assertNotIn("-e", cmd)
        self.assertNotIn("/sys/fs/cgroup:/sys/fs/cgroup:ro", cmd)
        self.assertNotIn("/models/custom:/models/custom", cmd)
        self.assertNotIn("NVIDIA_VISIBLE_DEVICES=all", cmd)

    def test_docker_run_omitted_volume_uses_current_model_path_default(self) -> None:
        response = prefill_command_web.build_command_response(
            {"model_path": "/models/custom"},
            profile="docker_run",
        )

        self.assertEqual(
            response["config"]["volume"],
            ["/sys/fs/cgroup:/sys/fs/cgroup:ro", "/models/custom:/models/custom"],
        )
        assert_option_block(
            self,
            response["command"],
            "-v",
            ["/sys/fs/cgroup:/sys/fs/cgroup:ro", "/models/custom:/models/custom"],
        )


class HandlerTests(unittest.TestCase):
    def _make_handler(self, method: str, path: str, body: bytes = b"") -> prefill_command_web.PrefillCommandHandler:
        handler = prefill_command_web.PrefillCommandHandler.__new__(prefill_command_web.PrefillCommandHandler)
        handler.path = path
        handler.command = method
        handler.request_version = "HTTP/1.1"
        handler.requestline = f"{method} {path} HTTP/1.1"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        return handler

    def _response(self, handler: prefill_command_web.PrefillCommandHandler) -> tuple[str, dict[str, str], bytes]:
        raw_headers, body = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
        lines = raw_headers.decode("iso-8859-1").split("\r\n")
        headers = dict(line.split(": ", 1) for line in lines[1:] if ": " in line)
        return lines[0], headers, body

    def _json_response(self, handler: prefill_command_web.PrefillCommandHandler) -> dict[str, object]:
        return json.loads(self._response(handler)[2].decode("utf-8"))

    def test_get_root_returns_static_html_without_inline_application(self) -> None:
        handler = self._make_handler("GET", "/")

        handler.do_GET()
        status_line, headers, body = self._response(handler)
        html = body.decode("utf-8")

        self.assertIn("200", status_line)
        self.assertEqual(headers["Content-Type"], "text/html; charset=utf-8")
        self.assertNotIn("Location", headers)
        self.assertIn('<form id="shared-model-form"', html)
        self.assertIn('<form id="docker-run-command-form"', html)
        self.assertIn('<form id="prefill-command-form"', html)
        self.assertIn('<form id="decode-command-form"', html)
        self.assertIn('<form id="router-command-form"', html)
        self.assertIn('data-profile-button="docker_run"', html)
        self.assertIn('data-profile-button="prefill"', html)
        self.assertIn('data-profile-button="decode"', html)
        self.assertIn('data-profile-button="router"', html)
        self.assertIn("SGLang Command Generator", html)
        self.assertIn("href=\"/styles.css\"", html)
        self.assertIn("src=\"/app.js\"", html)
        self.assertNotIn("<style>", html)
        self.assertNotIn("<script>\n", html)

    def test_get_static_css_and_js_assets(self) -> None:
        for path, content_type, expected in (
            ("/styles.css", "text/css; charset=utf-8", "--primary"),
            ("/app.js", "application/javascript; charset=utf-8", "fetch('/api/command'"),
        ):
            with self.subTest(path=path):
                handler = self._make_handler("GET", path)

                handler.do_GET()
                status_line, headers, body = self._response(handler)

                self.assertIn("200", status_line)
                self.assertEqual(headers["Content-Type"], content_type)
                self.assertIn(expected, body.decode("utf-8"))

    def test_unknown_static_path_returns_404(self) -> None:
        handler = self._make_handler("GET", "/missing.js")

        handler.do_GET()
        status_line, headers, body = self._response(handler)

        self.assertIn("404", status_line)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertIn("not found", body.decode("utf-8"))

    def test_get_defaults_returns_default_config(self) -> None:
        handler = self._make_handler("GET", "/api/defaults")

        handler.do_GET()
        body = self._json_response(handler)

        self.assertEqual(body["defaults"]["model_path"], "/mnt/GLM-5.1-FP8")
        self.assertEqual(body["defaults"]["tool_call_parser"], "glm47")
        self.assertEqual(body["defaults"]["reasoning_parser"], "glm45")
        self.assertEqual(prefill_command_web.DEFAULTS["tool_call_parser"], "unknown")
        self.assertEqual(prefill_command_web.DEFAULTS["reasoning_parser"], "unknown")
        self.assertEqual(body["defaults"]["mem_fraction_static"], 0.9)
        self.assertEqual(body["defaults"]["NCCL_IB_GID_INDEX"], "3")
        self.assertIn("unknown", body["parser_metadata"]["tool_call_parser_choices"])
        self.assertIn("unknown", body["parser_metadata"]["reasoning_parser_choices"])
        self.assertIn("glm47", body["parser_metadata"]["tool_call_parser_choices"])
        self.assertIn("glm45", body["parser_metadata"]["reasoning_parser_choices"])
        self.assertEqual(body["parser_metadata"]["fallbacks"]["tool_call_parser"], "unknown")
        self.assertEqual(body["parser_metadata"]["fallbacks"]["reasoning_parser"], "unknown")

    def test_get_defaults_supports_decode_router_and_docker_run_profiles(self) -> None:
        profile_expectations = (
            ("decode", "port", 30001),
            ("router", "port", 8000),
            ("docker_run", "image", "lmsysorg/sglang:latest"),
        )
        for profile, key, expected_value in profile_expectations:
            with self.subTest(profile=profile):
                handler = self._make_handler("GET", f"/api/defaults?profile={profile}")

                handler.do_GET()
                body = self._json_response(handler)

                self.assertEqual(body["profile"], profile)
                self.assertEqual(body["defaults"][key], expected_value)
                if profile == "docker_run":
                    self.assertEqual(body["defaults"]["model_path"], "/mnt/GLM-5.1-FP8")
                    self.assertEqual(
                        body["defaults"]["volume"],
                        "/sys/fs/cgroup:/sys/fs/cgroup:ro\n/mnt/GLM-5.1-FP8:/mnt/GLM-5.1-FP8",
                    )
                    self.assertEqual(body["defaults"]["env"], "NVIDIA_VISIBLE_DEVICES=all")
                    self.assertNotIn("docker_arg", body["defaults"])

    def test_get_defaults_rejects_unknown_profile(self) -> None:
        handler = self._make_handler("GET", "/api/defaults?profile=bad")

        handler.do_GET()
        response = handler.wfile.getvalue().decode("utf-8")

        self.assertIn("400", response.splitlines()[0])
        self.assertIn("unsupported profile", response)

    def test_post_api_command_returns_shell_command(self) -> None:
        handler = self._make_handler(
            "POST",
            "/api/command",
            json.dumps({"model_path": "/models/custom"}).encode("utf-8"),
        )

        handler.do_POST()
        body = self._json_response(handler)

        self.assertIn("shell_command", body)
        self.assertIn("--model-path /models/custom", body["shell_command"])
        self.assertFalse(body["executed"])

    def test_post_api_command_supports_router_profile(self) -> None:
        handler = self._make_handler(
            "POST",
            "/api/command",
            json.dumps({"profile": "router", "prefill": "http://p:30000", "decode": "http://d:30001"}).encode("utf-8"),
        )

        handler.do_POST()
        body = self._json_response(handler)

        self.assertEqual(body["profile"], "router")
        self.assertIn("sglang_router.launch_router", body["shell_command"])
        self.assertIn("--prefill http://p:30000", body["shell_command"])
        self.assertIn("--decode http://d:30001", body["shell_command"])

    def test_post_api_command_supports_docker_run_profile(self) -> None:
        handler = self._make_handler(
            "POST",
            "/api/command",
            json.dumps({"profile": "docker_run", "image": "custom/sglang:test"}).encode("utf-8"),
        )

        handler.do_POST()
        body = self._json_response(handler)

        self.assertEqual(body["profile"], "docker_run")
        self.assertIn("docker run", body["shell_command"])
        self.assertIn("custom/sglang:test", body["shell_command"])
        self.assertIn("-v /mnt/GLM-5.1-FP8:/mnt/GLM-5.1-FP8", body["shell_command"])
        self.assertFalse(body["executed"])

    def test_invalid_json_returns_400(self) -> None:
        handler = self._make_handler("POST", "/api/command", b"{")

        handler.do_POST()
        response = handler.wfile.getvalue().decode("utf-8")

        self.assertIn("400", response.splitlines()[0])
        self.assertIn("invalid JSON", response)

    def test_invalid_extra_args_returns_400(self) -> None:
        handler = self._make_handler(
            "POST",
            "/api/command",
            json.dumps({"extra_sglang_args": "--unterminated '"}).encode("utf-8"),
        )

        handler.do_POST()
        response = handler.wfile.getvalue().decode("utf-8")

        self.assertIn("400", response.splitlines()[0])
        self.assertIn("No closing quotation", response)


class WebUiStaticTests(unittest.TestCase):
    def test_mtp_fields_are_hidden_until_checkbox_is_enabled(self) -> None:
        html = Path("web/index.html").read_text(encoding="utf-8")
        css = Path("web/styles.css").read_text(encoding="utf-8")
        js = Path("web/app.js").read_text(encoding="utf-8")

        self.assertIn('name="enable_mtp"', html)
        self.assertNotIn('name="enable_mtp" checked', html)
        self.assertIn('data-mtp-fields hidden', html)
        self.assertRegex(css, r"\[hidden\]\s*\{\s*display:\s*none\s*!important;\s*\}")
        self.assertIn("mtpFields.hidden = !enabled;", js)
        self.assertIn("element.disabled = !enabled", js)
        self.assertIn("if (event.target.name === 'enable_mtp') updateMtpVisibility(profile);", js)

    def test_parallel_section_uses_radio_groups_and_hides_context_backend_until_selected(self) -> None:
        html = Path("web/index.html").read_text(encoding="utf-8")
        js = Path("web/app.js").read_text(encoding="utf-8")
        css = Path("web/styles.css").read_text(encoding="utf-8")

        self.assertIn('id="prefill-parallel-heading"', html)
        self.assertIn('world size', html)
        self.assertNotIn('TP / EP / PP size', html)
        self.assertIn('name="parallel_tp_size"', html)
        self.assertIn('data-dp-size-field hidden', html)
        self.assertIn('name="dp_size"', html)
        self.assertIn('name="attention_parallel_mode"', html)
        self.assertIn('value="dp_attention"', html)
        self.assertIn('value="context_parallel"', html)
        self.assertIn('value="pipeline_parallel"', html)
        self.assertIn('class="attention-option-stack"', html)
        self.assertIn('name="context_parallel_backend"', html)
        self.assertIn('data-context-backend-options hidden', html)
        self.assertIn('name="moe_parallel_mode"', html)
        self.assertIn('data-expert-overlap-options hidden', html)
        self.assertIn('name="enable_single_batch_overlap"', html)
        self.assertIn('name="enable_two_batch_overlap"', html)
        self.assertIn("updateContextBackendVisibility", js)
        self.assertIn("updateDpSizeVisibility", js)
        self.assertIn("dpSizeInput.value = worldSizeInput.value || defaults.parallel_tp_size || '';", js)
        self.assertIn("updateExpertOverlapVisibility", js)
        self.assertIn("if (!visible) element.checked = false;", js)
        self.assertIn(".context-backend-options .radio-option", css)
        self.assertIn(".context-backend-options,\n.pipeline-options,\n.dp-size-field,\n.expert-overlap-options {\n  padding-top: 8px;", css)
        self.assertNotIn("tensor_parallel_size", js)

    def test_hidden_fixed_fields_and_extra_args_are_not_rendered(self) -> None:
        html = Path("web/index.html").read_text(encoding="utf-8")
        js = Path("web/app.js").read_text(encoding="utf-8")

        for removed_text in (
            "Number of nodes",
            "Node rank",
            "Extra SGLang args",
            "NCCL IB GID index",
            "NCCL IB TC",
            "NCCL IB timeout",
            "NCCL IB retry count",
        ):
            with self.subTest(removed_text=removed_text):
                self.assertNotIn(removed_text, html)
                self.assertNotIn(removed_text, js)
        self.assertNotIn('name="extra_sglang_args"', html)
        self.assertNotIn('id="env-fields"', html)
        self.assertNotIn("extra_sglang_args", js)
        self.assertNotIn("NCCL_IB_GID_INDEX", js)
        self.assertNotIn("NCCL_IB_TC", js)
        self.assertNotIn("NCCL_IB_TIMEOUT", js)
        self.assertNotIn("NCCL_IB_RETRY_CNT", js)
        self.assertNotIn("{ name: 'disaggregation_mode', label: 'Mode' }", js)
        self.assertNotIn('name="disaggregation_mode"', html)
        self.assertIn("syncModelDerivedDefaults", js)
        self.assertIn("syncModelDerivedDefaults(sharedModelForm.querySelector('input[name=\"model_path\"]')?.value || defaults.model_path || '', profile);", js)
        self.assertIn("identifyButton.textContent = '识别模型';", js)
        self.assertIn("event.target.closest('[data-identify-model]')", js)
        self.assertIn("const modelPathInput = sharedModelForm.querySelector('input[name=\"model_path\"]');", js)
        self.assertIn("syncModelDerivedDefaults(modelPathInput?.value || '', state.activeProfile);", js)
        self.assertIn("if (event.target.name === 'model_path') {", js)
        self.assertIn("syncModelDerivedDefaults(event.target.value, state.activeProfile);", js)
        self.assertIn("if (state.activeProfile === 'docker_run') syncDockerRunModelVolume(event.target.value);", js)

    def test_pipeline_options_are_hidden_until_pipeline_parallel_selected(self) -> None:
        html = Path("web/index.html").read_text(encoding="utf-8")
        js = Path("web/app.js").read_text(encoding="utf-8")

        self.assertIn('data-pipeline-options hidden', html)
        self.assertIn('name="enable_dynamic_chunking"', html)
        self.assertIn('name="dynamic_chunking_smooth_factor"', html)
        self.assertIn('SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR', html)
        self.assertIn("updatePipelineOptionsVisibility", js)
        self.assertIn("pipelineOptions.hidden = !visible;", js)
        self.assertIn("element.disabled = !visible;", js)

    def test_shared_model_panel_is_above_profile_switcher(self) -> None:
        html = Path("web/index.html").read_text(encoding="utf-8")
        js = Path("web/app.js").read_text(encoding="utf-8")
        css = Path("web/styles.css").read_text(encoding="utf-8")

        self.assertIn('id="shared-model-form"', html)
        self.assertIn('data-model-fields', html)
        self.assertLess(html.index('id="shared-model-form"'), html.index('class="profile-switcher"'))
        self.assertLess(html.index('class="profile-switcher"'), html.index('id="prefill-command-form"'))
        self.assertIn("const sharedModelFields", js)
        for field_name in ("model_path", "served_model_name", "tool_call_parser", "reasoning_parser"):
            with self.subTest(field_name=field_name):
                self.assertIn(f"name: '{field_name}'", js)
        self.assertIn("sharedModelFieldsTarget?.replaceChildren(...sharedModelFields.map((field) => createField(field, 'shared')));", js)
        self.assertIn(".model-panel", css)
        self.assertIn(".shared-model-grid", css)

    def test_prefill_decode_router_are_independent_profile_forms(self) -> None:
        html = Path("web/index.html").read_text(encoding="utf-8")
        js = Path("web/app.js").read_text(encoding="utf-8")
        css = Path("web/styles.css").read_text(encoding="utf-8")

        self.assertIn('id="prefill-command-form"', html)
        self.assertIn('id="decode-command-form"', html)
        self.assertIn('id="router-command-form"', html)
        for profile in ("prefill", "decode", "router"):
            with self.subTest(profile=profile):
                form_start = html.index(f'id="{profile}-command-form"')
                form_end = html.index("</form>", form_start)
                form_html = html[form_start:form_end]
                self.assertNotIn(f'data-profile-fields="{profile}:model"', form_html)
                self.assertNotIn(f'{profile}-model-heading', form_html)
        self.assertIn('data-profile-button="prefill"', html)
        self.assertIn('data-profile-button="decode"', html)
        self.assertIn('data-profile-button="router"', html)
        self.assertIn("const profileConfigs", js)
        self.assertIn("const payload = { profile: state.activeProfile };", js)
        self.assertIn("const payloadForms = activeUsesSharedModel() ? [sharedModelForm, activeForm()] : [activeForm()];", js)
        self.assertIn("for (const element of form.elements)", js)
        self.assertIn("switchProfile", js)
        self.assertIn("/api/defaults?profile=", js)
        self.assertIn("applyModelDefaults(profile);", js)
        self.assertIn(".profile-switcher", css)
        self.assertIn(".profile-button.active", css)

    def test_prefill_and_decode_hide_trust_remote_code_and_decode_shows_disable_cuda_graph(self) -> None:
        html = Path("web/index.html").read_text(encoding="utf-8")

        def form_fragment(profile: str) -> str:
            form_start = html.index(f'id="{profile}-command-form"')
            form_end = html.index("</form>", form_start)
            return html[form_start:form_end]

        self.assertNotIn("Trust remote code", form_fragment("prefill"))
        self.assertNotIn('name="trust_remote_code"', form_fragment("prefill"))
        self.assertNotIn("Trust remote code", form_fragment("decode"))
        self.assertNotIn('name="trust_remote_code"', form_fragment("decode"))
        self.assertIn("Disable CUDA graph", form_fragment("prefill"))
        self.assertIn("Disable CUDA graph", form_fragment("decode"))
        self.assertIn('name="disable_cuda_graph"', form_fragment("decode"))

    def test_decode_limits_section_is_decode_only(self) -> None:
        html = Path("web/index.html").read_text(encoding="utf-8")
        js = Path("web/app.js").read_text(encoding="utf-8")

        def form_fragment(profile: str) -> str:
            form_start = html.index(f'id="{profile}-command-form"')
            form_end = html.index("</form>", form_start)
            return html[form_start:form_end]

        self.assertIn('id="decode-limits-heading">Limits</h3>', html)
        self.assertIn('data-profile-fields="decode:limits"', form_fragment("decode"))
        self.assertNotIn('data-profile-fields="prefill:limits"', form_fragment("decode"))
        self.assertNotIn('data-profile-fields="decode:limits"', form_fragment("prefill"))
        self.assertNotIn('data-profile-fields="decode:limits"', form_fragment("router"))
        self.assertIn("limits: [\n        { name: 'max_running_requests', label: 'Max running requests', type: 'number' },\n        { name: 'cuda_graph_max_bs', label: 'Cuda graph max bs', type: 'number' },\n      ],", js)
        self.assertEqual(js.count("name: 'max_running_requests'"), 1)
        self.assertEqual(js.count("name: 'cuda_graph_max_bs'"), 1)

    def test_docker_run_tab_is_first_and_has_independent_form(self) -> None:
        html = Path("web/index.html").read_text(encoding="utf-8")
        js = Path("web/app.js").read_text(encoding="utf-8")
        css = Path("web/styles.css").read_text(encoding="utf-8")

        self.assertLess(html.index('data-profile-button="docker_run"'), html.index('data-profile-button="prefill"'))
        self.assertLess(html.index('data-profile-button="prefill"'), html.index('data-profile-button="decode"'))
        self.assertLess(html.index('data-profile-button="decode"'), html.index('data-profile-button="router"'))
        self.assertIn('id="docker-run-command-form"', html)
        self.assertIn('data-profile-panel="docker_run"', html)
        self.assertIn('data-profile-fields="docker_run:container"', html)
        self.assertIn('data-profile-fields="docker_run:extra"', html)
        self.assertIn('name="rm"', html)
        for field_name in ("image", "container_name", "shm_size", "volume", "env"):
            with self.subTest(field_name=field_name):
                self.assertIn(f"name: '{field_name}'", js)
        self.assertNotIn("name: 'docker_arg'", js)
        self.assertNotIn("Extra Docker args", html)
        self.assertNotIn("Extra Docker args", js)
        self.assertNotIn("usesSharedModel: false", js)
        self.assertIn("usesSharedModel: true", js)
        self.assertIn("sharedModelForm.hidden = !activeUsesSharedModel();", js)
        self.assertIn("const payloadForms = activeUsesSharedModel() ? [sharedModelForm, activeForm()] : [activeForm()];", js)
        self.assertNotIn("appendDockerRunModelVolume(payload);", js)
        self.assertIn("syncDockerRunModelVolume", js)
        self.assertIn("const nextModelVolume = modelVolume ? `${modelVolume}:${modelVolume}` : '';", js)
        self.assertIn("switchProfile('docker_run', { refresh: false });", js)
        self.assertIn("heading.className = 'list-field-heading';", js)
        self.assertIn("heading.append(labelText, addButton);", js)
        self.assertIn("data-add-list-item", js)
        self.assertIn("data-delete-list-item", js)
        self.assertIn("addButton.textContent = '添加';", js)
        self.assertIn("deleteButton.textContent = '删除';", js)
        self.assertIn("setListFieldItems(field, [...getListFieldItems(field), '']);", js)
        self.assertIn("input.dataset.listItemInput = field;", js)
        self.assertIn("syncListFieldValue(listInput.dataset.listItemInput);", js)
        self.assertIn("deleteButton.dataset.index = String(index);", js)
        self.assertIn("deleteListFieldItem(deleteListButton.dataset.deleteListItem, deleteListButton.dataset.index);", js)
        self.assertNotIn("data-list-input", js)
        self.assertNotIn("inputRow.append(input, addButton);", js)
        self.assertNotIn("input.dataset.samePathMapping = 'true';", js)
        self.assertNotIn("const value = input.dataset.samePathMapping === 'true' ? `${rawValue}:${rawValue}` : rawValue;", js)
        self.assertIn(".docker-run-grid", css)
        self.assertIn(".list-field-heading", css)
        self.assertIn(".delete-list-item-button", css)

    def test_docker_run_ui_does_not_expose_fixed_default_docker_args(self) -> None:
        html = Path("web/index.html").read_text(encoding="utf-8")
        js = Path("web/app.js").read_text(encoding="utf-8")

        for fixed_arg in (
            "--user=0",
            "--privileged",
            "--ipc=host",
            "--network host",
            "--runtime=nvidia",
            "--gpus all",
            "--ulimit memlock=-1:-1",
            "--entrypoint /bin/bash",
        ):
            with self.subTest(fixed_arg=fixed_arg):
                self.assertNotIn(fixed_arg, html)
                self.assertNotIn(fixed_arg, js)


class ReadmeDocumentationTests(unittest.TestCase):
    def test_readme_documents_parallel_tp_size_without_old_tensor_parallel_field(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("--tp-size 8", readme)
        self.assertIn('"parallel_tp_size":4', readme)
        self.assertIn("SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR", readme)
        self.assertIn("默认启用 / `0.8`", readme)
        self.assertNotIn("--tensor-parallel-size", readme)
        self.assertNotIn("tensor_parallel_size", readme)


class GitignoreTests(unittest.TestCase):
    def test_hidden_directories_are_ignored_while_gitignore_stays_tracked(self) -> None:
        patterns = Path(".gitignore").read_text(encoding="utf-8").splitlines()

        self.assertIn(".*/", patterns)
        self.assertIn("!.gitignore", patterns)


class MainSafetyTests(unittest.TestCase):
    def test_module_does_not_import_subprocess_or_expose_execution_endpoint(self) -> None:
        with open(prefill_command_web.__file__, encoding="utf-8") as source_file:
            source = source_file.read()

        self.assertNotIn("import subprocess", source)
        self.assertNotIn("subprocess.run", source)
        self.assertNotIn("shell=True", source)
        self.assertNotIn("/start", source)
        self.assertNotIn("/run", source)
        self.assertNotIn("execute=true", source)

    def test_default_cli_port_is_6060(self) -> None:
        args = prefill_command_web.parse_args([])

        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 6060)

    def test_main_serves_default_6060_web_ui_without_running_prefill(self) -> None:
        with mock.patch.object(prefill_command_web, "run_server") as run_server_mock:
            exit_code = prefill_command_web.main([])

        self.assertEqual(exit_code, 0)
        run_server_mock.assert_called_once_with("127.0.0.1", 6060)

    def test_main_accepts_explicit_port_override(self) -> None:
        with mock.patch.object(prefill_command_web, "run_server") as run_server_mock:
            exit_code = prefill_command_web.main(["--host", "127.0.0.1", "--port", "9090"])

        self.assertEqual(exit_code, 0)
        run_server_mock.assert_called_once_with("127.0.0.1", 9090)


if __name__ == "__main__":
    unittest.main()
