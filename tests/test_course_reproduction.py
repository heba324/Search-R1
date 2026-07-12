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
        from scripts.course_reproduction.preflight import HostInfo, assess_host

        self.assertEqual(assess_host(HostInfo(1, 80, 120, 500)), [])
        message = "\n".join(assess_host(HostInfo(1, 40, 64, 200)))
        self.assertIn("80 GiB", message)
        self.assertIn("120 GiB RAM", message)
        self.assertIn("500 GiB", message)

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
        self.assertIn("TOTAL_STEPS=10", timing)
        self.assertIn("TRAIN_BATCH_SIZE=32", timing)
        self.assertIn("VAL_BEFORE_TRAIN=false", timing)
        self.assertIn("VAL_DATA_NUM=7", timing)

    def test_bm25_assets_and_server_are_cpu_only(self):
        prepare = self.read_script("prepare_bm25_index.sh")
        launch = self.read_script("launch_bm25_retriever.sh")
        self.assertIn("PeterJinGo/wiki-18-bm25-index", prepare)
        self.assertIn("2c7554f", prepare)
        self.assertIn("bm25_server.py", launch)
        self.assertIn("--topk 3", launch)
        self.assertNotIn("--faiss_gpu", launch)
        self.assertIn("CUDA_VISIBLE_DEVICES=", launch)

    def test_setup_installs_cpu_bm25_runtime(self):
        script = self.read_script("setup_envs.sh")
        self.assertIn("openjdk=21", script)
        self.assertIn("pyserini==0.25.0", script)
        self.assertIn("torch==2.4.0", script)
        self.assertIn("vllm==0.6.3", script)

    def test_model_download_is_revision_pinned(self):
        script = self.read_script("prepare_model.sh")
        self.assertIn("Qwen/Qwen2.5-1.5B-Instruct", script)
        self.assertIn("989aa79", script)

    def test_fixed_eval_subset_is_balanced_and_deterministic(self):
        from scripts.course_reproduction.prepare_eval_data import select_indices

        self.assertEqual(select_indices(10, 4, 42), select_indices(10, 4, 42))
        self.assertEqual(len(select_indices(10, 4, 42)), 4)
        self.assertNotEqual(select_indices(10, 4, 42), select_indices(10, 4, 43))

    def test_evaluation_supports_pre_and_post_rl_models(self):
        script = self.read_script("evaluate.sh")
        self.assertIn('MODEL_PATH="${MODEL_PATH:-', script)
        self.assertIn('EVAL_RUN_NAME="${EVAL_RUN_NAME:-post-rl}"', script)
        self.assertIn("data/course_eval/test.parquet", script)
        self.assertIn("+trainer.val_only=true", script)
        self.assertIn("algorithm.adv_estimator=grpo", script)

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
        self.assertIn('json.loads(raw)["contents"]', schema)
        self.assertIn('@app.post("/retrieve")', server)

    def test_bm25_server_preserves_search_r1_document_schema(self):
        from scripts.course_reproduction.bm25_schema import document_from_raw

        document = document_from_raw('{"contents": "\\\"Beijing\\\"\\nCapital of China."}')
        self.assertEqual(document["title"], "Beijing")
        self.assertEqual(document["text"], "Capital of China.")
        self.assertEqual(document["contents"], '"Beijing"\nCapital of China.')


if __name__ == "__main__":
    unittest.main()
