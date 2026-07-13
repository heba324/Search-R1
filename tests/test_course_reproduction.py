import unittest
import subprocess
from pathlib import Path


class CourseReproductionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.script_root = cls.repo_root / "scripts" / "course_reproduction"

    def read_script(self, name):
        return (self.script_root / name).read_text(encoding="utf-8")

    def test_contract_records_resource_limited_method_reproduction(self):
        from scripts.course_reproduction.contract import COURSE_REPRODUCTION

        self.assertEqual(COURSE_REPRODUCTION.paper_version, "arXiv v5 / COLM 2025")
        self.assertEqual(
            COURSE_REPRODUCTION.baseline_commit,
            "eaa5a66c0ed779d195a4bd0e165f0c73b99f12de",
        )
        self.assertEqual(COURSE_REPRODUCTION.model_id, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(COURSE_REPRODUCTION.algorithm, "grpo")
        self.assertEqual(COURSE_REPRODUCTION.retriever, "bm25")
        self.assertEqual(COURSE_REPRODUCTION.training_steps, 120)
        self.assertEqual(COURSE_REPRODUCTION.train_batch_size, 32)
        self.assertEqual(COURSE_REPRODUCTION.group_size, 5)
        self.assertEqual(COURSE_REPRODUCTION.max_turns, 4)
        self.assertEqual(COURSE_REPRODUCTION.topk, 3)
        self.assertEqual(COURSE_REPRODUCTION.eval_examples_per_dataset, 100)
        self.assertEqual(COURSE_REPRODUCTION.seed, 42)

    def test_course_host_requires_one_large_gpu(self):
        from scripts.course_reproduction.preflight import MIN_DISK_GIB, MIN_RAM_GIB, HostInfo, assess_host

        self.assertEqual(MIN_RAM_GIB, 110)
        self.assertEqual(MIN_DISK_GIB, 420)
        self.assertEqual(assess_host(HostInfo(1, 80, 110, 420, 32)), [])
        message = "\n".join(assess_host(HostInfo(1, 40, 109, 419, 31)))
        self.assertIn("80 GiB", message)
        self.assertIn("110 GiB RAM", message)
        self.assertIn("420 GiB", message)
        self.assertIn("32 GiB /dev/shm", message)

    def test_host_preflight_requires_cloud_toolchain(self):
        script = self.read_script("preflight.py")
        self.assertIn('("conda", "git", "nvidia-smi", "nvcc", "tmux")', script)
        self.assertIn("merge-base", script)
        self.assertIn("baseline_commit", script)

    def test_training_cli_defaults_to_single_gpu_grpo(self):
        script = self.read_script("train_grpo.sh")
        for expected in (
            "Qwen2.5-1.5B-Instruct",
            'TOTAL_STEPS="${TOTAL_STEPS:-120}"',
            'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"',
            "algorithm.adv_estimator=grpo",
            'GROUP_SIZE="${GROUP_SIZE:-5}"',
            'actor_rollout_ref.rollout.n_agent="$GROUP_SIZE"',
            "trainer.n_gpus_per_node=1",
            "data/course_eval/test.parquet",
            'TEST_FREQ="${TEST_FREQ:-50}"',
            'ENGINE_STOP_STEP="$(( TOTAL_STEPS + 1 ))"',
            'trainer.total_training_steps="$ENGINE_STOP_STEP"',
            "max_turns=4",
            "retriever.topk=3",
        ):
            self.assertIn(expected, script)
        self.assertNotIn("critic.model.path", script)

    def test_smoke_and_timing_commands_are_fixed_and_cheap(self):
        smoke = self.read_script("run_smoke.sh")
        timing = self.read_script("run_timing.sh")
        self.assertIn("TOTAL_STEPS=2", smoke)
        self.assertIn("TRAIN_BATCH_SIZE=8", smoke)
        self.assertIn("VAL_BEFORE_TRAIN=false", smoke)
        self.assertIn("VAL_DATA_NUM=7", smoke)
        self.assertIn("VAL_BATCH_SIZE=7", smoke)
        self.assertIn("TOTAL_STEPS=10", timing)
        self.assertIn("TRAIN_BATCH_SIZE=32", timing)
        self.assertIn("VAL_BEFORE_TRAIN=false", timing)
        self.assertIn("VAL_DATA_NUM=7", timing)
        self.assertIn("VAL_BATCH_SIZE=7", timing)

    def test_bm25_assets_and_server_are_cpu_only(self):
        prepare = self.read_script("prepare_bm25_index.sh")
        launch = self.read_script("launch_bm25_retriever.sh")
        self.assertIn("PeterJinGo/wiki-18-bm25-index", prepare)
        self.assertIn("2c7554f", prepare)
        self.assertIn("PeterJinGo/wiki-18-corpus", prepare)
        self.assertIn("69c1c00", prepare)
        self.assertIn("bm25_server.py", launch)
        self.assertIn("--topk 3", launch)
        self.assertIn("--corpus-path", launch)
        self.assertNotIn("--faiss_gpu", launch)
        self.assertIn("CUDA_VISIBLE_DEVICES=", launch)

    def test_setup_installs_cpu_bm25_runtime(self):
        script = self.read_script("setup_envs.sh")
        self.assertIn("openjdk=17", script)
        self.assertIn("pyserini==0.25.0", script)
        self.assertIn("torch==2.4.0", script)
        self.assertIn("vllm==0.6.3", script)
        self.assertIn("PIP_CACHE_DIR", script)

    def test_bm25_launcher_pins_java_17_after_conda_activation(self):
        script = self.read_script("launch_bm25_retriever.sh")
        activate = script.index('conda activate "$RETRIEVER_ENV"')
        java_home = script.index('JAVA_HOME="$CONDA_PREFIX"')
        self.assertLess(activate, java_home)
        self.assertIn('PATH="$JAVA_HOME/bin:$PATH"', script)
        self.assertIn("Expected Java 17", script)

    def test_large_downloads_and_dataset_cache_stay_on_workspace_disk(self):
        for name in ("prepare_model.sh", "prepare_bm25_index.sh", "launch_bm25_retriever.sh"):
            self.assertIn("HF_HOME", self.read_script(name), name)
        eval_script = self.read_script("prepare_eval_data.py")
        self.assertIn("HF_DATASETS_CACHE", eval_script)

    def test_model_download_is_revision_pinned(self):
        script = self.read_script("prepare_model.sh")
        self.assertIn("Qwen/Qwen2.5-1.5B-Instruct", script)
        self.assertIn("989aa79", script)

    def test_fixed_eval_subset_is_balanced_and_deterministic(self):
        from scripts.course_reproduction.prepare_eval_data import select_indices

        self.assertEqual(select_indices(10, 4, 42), select_indices(10, 4, 42))
        self.assertEqual(len(select_indices(10, 4, 42)), 4)
        self.assertNotEqual(select_indices(10, 4, 42), select_indices(10, 4, 43))
        with self.assertRaises(ValueError):
            select_indices(3, 4, 42)

    def test_evaluation_supports_pre_and_post_rl_models(self):
        script = self.read_script("evaluate.sh")
        self.assertIn('MODEL_PATH="${MODEL_PATH:-', script)
        self.assertIn('EVAL_RUN_NAME="${EVAL_RUN_NAME:-post-rl}"', script)
        self.assertIn("data/course_eval/test.parquet", script)
        self.assertIn("+trainer.val_only=true", script)
        self.assertIn("algorithm.adv_estimator=grpo", script)
        self.assertIn('--elapsed-seconds "$ELAPSED"', script)
        self.assertIn('--eval-data "$EVAL_DATA"', script)
        self.assertIn('--model-path "$MODEL_PATH"', script)
        self.assertIn("compare_evaluations.py", script)
        self.assertIn("main_ppo_with_behavior", script)

    def test_search_behavior_summary_counts_attempts_and_duplicates(self):
        from scripts.course_reproduction.search_behavior import summarize_search_behavior

        summary = summarize_search_behavior(
            "<think>x</think><search>Alpha  Beta</search>"
            "<information><search>document markup</search></information>"
            "<search> alpha beta </search>"
            "<search>   </search><answer>done</answer>"
        )
        self.assertEqual(summary["searches"], 3)
        self.assertEqual(summary["valid_searches"], 2)
        self.assertEqual(summary["duplicate_searches"], 1)
        self.assertEqual(summary["invalid_searches"], 1)
        self.assertEqual(summary["used_search"], 1)

    def test_eval_parser_captures_search_behavior(self):
        from scripts.course_reproduction.parse_eval_metrics import parse_search_behavior

        parsed = parse_search_behavior(
            "{'val/search_behavior/avg_searches/overall': 1.25, "
            "'val/search_behavior/search_rate/overall': 0.75}"
        )
        self.assertEqual(parsed["overall"]["avg_searches"], 1.25)
        self.assertEqual(parsed["overall"]["search_rate"], 0.75)

    def test_pre_post_comparison_records_em_and_behavior_deltas(self):
        from scripts.course_reproduction.compare_evaluations import compare_payloads

        before = {
            "metrics": {"nq": 0.2},
            "search_behavior": {"nq": {"avg_searches": 2.0}},
            "evaluation_data_sha256": "same-data",
        }
        after = {
            "metrics": {"nq": 0.3},
            "search_behavior": {"nq": {"avg_searches": 1.5}},
            "evaluation_data_sha256": "same-data",
        }
        comparison = compare_payloads(before, after)
        self.assertAlmostEqual(comparison["em_delta"]["nq"], 0.1)
        self.assertEqual(
            comparison["search_behavior_delta"]["nq"]["avg_searches"], -0.5
        )
        after["evaluation_data_sha256"] = "different-data"
        with self.assertRaises(ValueError):
            compare_payloads(before, after)

    def test_course_workflow_uses_course_retriever_gate(self):
        for name in ("train_grpo.sh", "evaluate.sh"):
            self.assertIn("scripts/course_reproduction/check_retriever.py", self.read_script(name), name)

    def test_resource_runbook_states_scientific_scope(self):
        doc = (self.repo_root / "docs" / "course_reproduction_zh.md").read_text(encoding="utf-8")
        for expected in (
            "方法复现",
            "不是论文表格的严格数值复现",
            "1×A800 80GB",
            "Qwen2.5-1.5B-Instruct",
            "GRPO",
            "BM25",
            "Pre-RL Baseline",
            "120 steps",
            "七个数据集",
        ):
            self.assertIn(expected, doc)
        self.assertIn("W&B step 121", doc)
        self.assertIn("6 × 单次七数据集评测秒数", doc)
        self.assertIn("943机", doc)
        self.assertIn("455GB", doc)
        self.assertIn("/root/autodl-tmp", doc)

    def test_course_shell_entrypoints_are_executable(self):
        output = subprocess.check_output(
            ["git", "ls-files", "--stage", "scripts/course_reproduction/*.sh"],
            cwd=self.repo_root,
            text=True,
        )
        modes = [line.split()[0] for line in output.splitlines()]
        self.assertTrue(modes)
        self.assertEqual(set(modes), {"100755"})

    def test_course_bm25_server_uses_official_pyserini_index(self):
        server = self.read_script("bm25_server.py")
        schema = self.read_script("bm25_schema.py")
        self.assertIn("LuceneSearcher", server)
        self.assertIn("load_dataset", server)
        self.assertIn("build_search_batches", server)
        self.assertIn("document_from_raw", server)
        self.assertIn("document_from_record(json.loads(raw))", schema)
        self.assertIn('@app.post("/retrieve")', server)

    def test_bm25_server_preserves_search_r1_document_schema(self):
        from scripts.course_reproduction.bm25_schema import (
            build_search_batches,
            document_from_raw,
            document_from_record,
        )

        document = document_from_raw('{"contents": "\\\"Beijing\\\"\\nCapital of China."}')
        self.assertEqual(document["title"], "Beijing")
        self.assertEqual(document["text"], "Capital of China.")
        self.assertEqual(document["contents"], '"Beijing"\nCapital of China.')
        self.assertEqual(document_from_record({"contents": '"Paris"\nCapital of France.'})["title"], "Paris")

        class Hit:
            def __init__(self, docid, score):
                self.docid = docid
                self.score = score

        def search(query, requested):
            if query == "explode":
                raise ValueError("query parser failure")
            if query == "short":
                return [Hit("only", 1.0)]
            return [Hit(str(index), float(index)) for index in range(requested)]

        result = build_search_batches(
            ["", "explode", "short", "normal"],
            requested=3,
            return_scores=True,
            search=search,
            document_for_hit=lambda hit: {
                "title": hit.docid,
                "text": "text",
                "contents": f'{hit.docid}\ntext',
            },
        )
        self.assertEqual(result["result"][0], [])
        self.assertEqual(result["result"][1], [])
        self.assertEqual(len(result["result"][2]), 1)
        self.assertEqual(len(result["result"][3]), 3)

    def test_evidence_archive_does_not_hash_or_pack_itself(self):
        script = self.read_script("collect_evidence.sh")
        self.assertIn("-path \"$OUT\" -prune", script)
        self.assertIn('TMP_ARCHIVE="$(mktemp', script)
        self.assertIn('rm -f "$OUT/course-reproduction-evidence.tar.gz"', script)


if __name__ == "__main__":
    unittest.main()
