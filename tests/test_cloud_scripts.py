from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


class CloudScriptContractTests(unittest.TestCase):
    def read(self, name: str) -> str:
        return (SCRIPTS / name).read_text(encoding="utf-8")

    def test_requested_conda_environment_names_are_defaults(self):
        training = self.read("cloud_setup_searchr1.sh")
        retriever = self.read("cloud_setup_retriever.sh")
        self.assertIn('ENV_NAME="${ENV_NAME:-Search-R1}"', training)
        self.assertIn('ENV_NAME="${ENV_NAME:-Search-R1-retriever}"', retriever)

    def test_all_launchers_resolve_repository_root(self):
        names = (
            "cloud_setup_searchr1.sh",
            "cloud_setup_retriever.sh",
            "cloud_prepare_smoke_assets.sh",
            "cloud_prepare_data_and_index.sh",
            "cloud_launch_retriever.sh",
            "cloud_train_grpo_smoke.sh",
            "cloud_train_grpo_full.sh",
        )
        for name in names:
            script = self.read(name)
            self.assertIn('REPO_ROOT=', script, name)
            self.assertIn('cd "$REPO_ROOT"', script, name)

    def test_smoke_assets_use_tracked_example_corpus(self):
        script = self.read("cloud_prepare_smoke_assets.sh")
        self.assertIn('example/corpus.jsonl', script)
        self.assertIn('search_r1/search/index_builder.py', script)
        self.assertIn('SAVE_PATH="${SAVE_PATH:-$REPO_ROOT/data/smoke_retriever}"', script)
        self.assertIn('INDEX_FILE="$SAVE_PATH/e5_Flat.index"', script)

    def test_retriever_has_smoke_and_full_asset_profiles(self):
        script = self.read("cloud_launch_retriever.sh")
        self.assertIn('ASSET_PROFILE="${ASSET_PROFILE:-smoke}"', script)
        self.assertIn('data/smoke_retriever/e5_Flat.index', script)
        self.assertIn('data/wiki18/e5_Flat.index', script)
        self.assertIn('artifacts/retriever_profile.txt', script)
        self.assertNotIn('PORT="${PORT', script)

    def test_smoke_run_calls_smoke_preflight(self):
        script = self.read("cloud_train_grpo_smoke.sh")
        self.assertIn('cloud_preflight.py --profile smoke', script)
        self.assertIn('SEARCH_ENV="${SEARCH_ENV:-Search-R1}"', script)

    def test_training_python_is_overridable_for_preflight_testing(self):
        for name in ("cloud_train_grpo_smoke.sh", "cloud_train_grpo_full.sh"):
            script = self.read(name)
            self.assertIn('PYTHON_BIN="${PYTHON_BIN:-python3}"', script, name)
            self.assertIn('"$PYTHON_BIN" scripts/cloud_preflight.py', script, name)

    def test_full_run_has_hardware_and_confirmation_gates(self):
        script = self.read("cloud_train_grpo_full.sh")
        self.assertIn('cloud_preflight.py --profile full', script)
        self.assertIn('CONFIRM_FULL_RUN', script)
        self.assertIn('CONFIRM_FULL_RUN must be YES', script)
        self.assertIn('artifacts/retriever_profile.txt', script)
        self.assertIn('full retriever profile', script)

    def test_console_logging_is_default(self):
        script = self.read("cloud_train_grpo_full.sh")
        self.assertIn('TRAINER_LOGGER="${TRAINER_LOGGER:-console}"', script)
        self.assertIn('trainer.logger=[\'${TRAINER_LOGGER}\']', script)

    def test_full_data_preparation_checks_all_outputs(self):
        script = self.read("cloud_prepare_data_and_index.sh")
        for expected in (
            'part_aa',
            'part_ab',
            'e5_Flat.index',
            'wiki-18.jsonl',
            'train.parquet',
            'test.parquet',
        ):
            self.assertIn(expected, script)

    def test_full_corpus_decompression_is_atomic(self):
        script = self.read("cloud_prepare_data_and_index.sh")
        self.assertIn('CORPUS_TMP=', script)
        self.assertIn('gzip -cd "$CORPUS_GZ" > "$CORPUS_TMP"', script)
        self.assertIn('mv "$CORPUS_TMP" "$CORPUS_FILE"', script)

    def test_evidence_script_records_reproduction_metadata(self):
        script = self.read("cloud_collect_evidence.sh")
        for expected in (
            'git rev-parse HEAD',
            'git status --short',
            'nvidia-smi',
            'free -h',
            'df -h',
            'conda list -n "$SEARCH_ENV" --explicit',
            'conda list -n "$RETRIEVER_ENV" --explicit',
            'sha256sum',
            'train.parquet',
            'test.parquet',
            '*.log',
            'verl_checkpoints',
        ):
            self.assertIn(expected, script)
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("artifacts/", gitignore)

    def test_beginner_docs_use_current_commands(self):
        paths = (
            ROOT / "docs" / "beginner_cloud_reproduction_zh.md",
            ROOT / "docs" / "rental_reproduction_runbook.md",
        )
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for expected in (
            "https://github.com/heba324/Search-R1",
            "Search-R1-retriever",
            "cloud_preflight.py --profile smoke",
            "cloud_prepare_smoke_assets.sh",
            "ASSET_PROFILE=full bash scripts/cloud_launch_retriever.sh",
            "CONFIRM_FULL_RUN=YES bash scripts/cloud_train_grpo_full.sh",
            "cloud_collect_evidence.sh",
        ):
            self.assertIn(expected, combined)
        for obsolete in (
            "conda activate searchr1",
            "conda activate retriever",
            "Search-R1-reproduce",
            "你的用户名",
        ):
            self.assertNotIn(obsolete, combined)

    def test_cloud_shell_scripts_are_executable_in_git(self):
        output = subprocess.check_output(
            ["git", "ls-files", "--stage", "scripts/cloud_*.sh"],
            cwd=ROOT,
            text=True,
        )
        entries = [line for line in output.splitlines() if line]
        self.assertGreater(len(entries), 0)
        for entry in entries:
            mode, _, _, path = entry.split(maxsplit=3)
            self.assertEqual(mode, "100755", path)


if __name__ == "__main__":
    unittest.main()
