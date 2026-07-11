import unittest
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.paper_v1.contract import (
    PAPER_V1,
    PAPER_V1_TARGET_EM,
    REQUIRED_EVALUATION_DATASETS,
    assert_paper_commit,
    assess_required_assets,
    assess_result_metrics,
)
from scripts.paper_v1.preflight import HostInfo, assess_author_source, assess_host
from scripts.paper_v1.prepare_eval_data import build_record
from scripts.paper_v1.parse_eval_metrics import parse_metrics
from scripts.paper_v1.check_retriever import validate_response


class PaperV1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]

    def read_wrapper(self, name):
        return (self.repo_root / "scripts" / "paper_v1" / name).read_text(encoding="utf-8")

    def test_frozen_paper_v1_constants(self):
        self.assertEqual(PAPER_V1.git_commit, "118c6e7361bb68e33c525b50d62f83b63462799e")
        self.assertEqual(PAPER_V1.model_id, "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(PAPER_V1.algorithm, "ppo")
        self.assertEqual(PAPER_V1.training_steps, 305)
        self.assertEqual(PAPER_V1.max_turns, 4)
        self.assertEqual(PAPER_V1.topk, 3)
        self.assertEqual(PAPER_V1.target_average_em, 0.327)
        self.assertEqual(PAPER_V1.dataset_revision, "b7d80abfee334a7a91cb377544f09180d58b34f6")
        self.assertEqual(PAPER_V1_TARGET_EM["nq"], 0.323)
        self.assertEqual(PAPER_V1_TARGET_EM["bamboogle"], 0.315)
        self.assertAlmostEqual(sum(PAPER_V1_TARGET_EM.values()) / 7, 0.327, places=3)

    def test_commit_guard_rejects_wrong_source_revision(self):
        with self.assertRaisesRegex(ValueError, "118c6e7"):
            assert_paper_commit("598e61bd1d36895726d28a8d06b3a15bed19f5d3")

    def test_asset_assessment_requires_published_train_and_test_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            errors = assess_required_assets(root)
        message = "\n".join(errors)
        self.assertIn("train.parquet", message)
        self.assertIn("test.parquet", message)
        self.assertIn("e5_Flat.index", message)
        self.assertIn("wiki-18.jsonl", message)

    def test_metric_assessment_requires_all_paper_datasets(self):
        errors = assess_result_metrics({"nq": 0.4})
        message = "\n".join(errors)
        self.assertIn("triviaqa", message)
        self.assertEqual(
            REQUIRED_EVALUATION_DATASETS,
            ("nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"),
        )

    def test_host_assessment_requires_paper_scale_resources(self):
        errors = assess_host(HostInfo(gpu_count=4, ram_gib=96, disk_gib=400))
        message = "\n".join(errors)
        self.assertIn("at least 8 NVIDIA GPUs", message)
        self.assertIn("128 GiB RAM", message)
        self.assertIn("500 GiB free disk", message)

    def test_current_branch_preserves_author_training_core(self):
        self.assertEqual(assess_author_source(self.repo_root), [])

    def test_training_wrapper_matches_v1_primary_row(self):
        script = self.read_wrapper("train_qwen25_3b_instruct_ppo.sh")
        for expected in (
            "Qwen/Qwen2.5-3B-Instruct",
            "algorithm.adv_estimator=gae",
            "trainer.total_training_steps=305",
            "max_turns=4",
            "retriever.topk=3",
            "actor_rollout_ref.rollout.n_agent=1",
            "data/nq_hotpotqa_train/train.parquet",
        ):
            self.assertIn(expected, script)
        self.assertNotIn("format_score", script)
        self.assertNotIn("adv_estimator=grpo", script)

    def test_data_wrapper_pins_published_dataset_revision_and_hashes(self):
        script = self.read_wrapper("prepare_train_data.sh")
        for expected in (
            "b7d80abfee334a7a91cb377544f09180d58b34f6",
            "c3cc21e862a8469105de666101578cbff23cdc77e91a803cef102622c89cc4f6",
            "30aa887b6d47e06e8c0f6f5307c88fe4e13461ac25a20ec0a5433ad7a4fe25dc",
            "355663891",
            "70370337",
        ):
            self.assertIn(expected, script)

    def test_evaluation_wrapper_requires_seven_dataset_parquet(self):
        script = self.read_wrapper("evaluate_qwen25_3b_instruct_ppo.sh")
        self.assertIn("data/paper_v1_eval/test.parquet", script)
        self.assertIn("actor/global_step_300", script)
        self.assertIn("+trainer.val_only=true", script)
        self.assertIn("max_turns=4", script)
        self.assertIn("retriever.topk=3", script)

    def test_eval_record_uses_v1_prompt_and_em_schema(self):
        record = build_record("triviaqa", "Who wrote Hamlet", ["William Shakespeare"], 7)
        self.assertIn("<search>", record["prompt"][0]["content"])
        self.assertTrue(record["prompt"][0]["content"].endswith("Who wrote Hamlet?\n"))
        self.assertEqual(record["reward_model"]["ground_truth"]["target"], ["William Shakespeare"])
        self.assertEqual(record["data_source"], "triviaqa")

    def test_setup_and_retrieval_wrappers_pin_required_versions(self):
        setup = self.read_wrapper("setup_envs.sh")
        retrieval = self.read_wrapper("prepare_retrieval_assets.sh")
        models = self.read_wrapper("prepare_models.sh")
        self.assertIn("conda create -y -n \"$SEARCH_ENV\" python=3.9", setup)
        self.assertIn("torch==2.4.0", setup)
        self.assertIn("vllm==0.6.3", setup)
        self.assertIn("faiss-gpu=1.8.0", setup)
        self.assertIn("a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023", retrieval)
        self.assertIn("b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11", retrieval)
        self.assertIn("7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db", retrieval)
        self.assertIn("Qwen/Qwen2.5-3B-Instruct", models)
        self.assertIn("aa8e725", models)
        self.assertIn("intfloat/e5-base-v2", models)
        self.assertIn("f52bf8e", models)

    def test_retriever_and_evidence_wrappers_record_paper_identity(self):
        launcher = self.read_wrapper("launch_retriever.sh")
        evidence = self.read_wrapper("collect_evidence.sh")
        self.assertIn("data/wiki18/e5_Flat.index", launcher)
        self.assertIn("data/wiki18/wiki-18.jsonl", launcher)
        self.assertIn("--topk 3", launcher)
        for expected in ("118c6e7", "nvidia-smi", "conda list", "pip freeze", "paper-v1.sha256"):
            self.assertIn(expected, evidence)

    def test_beginner_runbook_is_explicit_about_v1_target(self):
        doc = (self.repo_root / "docs" / "paper_v1_reproduction_zh.md").read_text(encoding="utf-8")
        for expected in (
            "arXiv v1",
            "118c6e7",
            "Qwen2.5-3B-Instruct",
            "PPO",
            "305 steps",
            "0.327",
            "NQ",
            "Bamboogle",
            "停止计费",
        ):
            self.assertIn(expected, doc)

    def test_metric_parser_extracts_all_scores_from_one_console_line(self):
        text = "step:0 - val/test_score/nq:0.323 - val/test_score/triviaqa:0.537"
        self.assertEqual(parse_metrics(text), {"nq": 0.323, "triviaqa": 0.537})

    def test_paper_v1_shell_wrappers_are_executable(self):
        output = subprocess.check_output(
            ["git", "ls-files", "--stage", "scripts/paper_v1/*.sh"],
            cwd=self.repo_root,
            text=True,
        )
        modes = [line.split()[0] for line in output.splitlines()]
        self.assertTrue(modes)
        self.assertEqual(set(modes), {"100755"})

    def test_retriever_check_rejects_incomplete_response(self):
        with self.assertRaisesRegex(ValueError, "3 documents"):
            validate_response({"result": [[{"document": {"contents": "one"}, "score": 1.0}]]}, expected=3)

    def test_training_and_evaluation_call_retriever_check(self):
        for name in ("train_qwen25_3b_instruct_ppo.sh", "evaluate_qwen25_3b_instruct_ppo.sh"):
            self.assertIn("check_retriever.py", self.read_wrapper(name), name)


if __name__ == "__main__":
    unittest.main()
