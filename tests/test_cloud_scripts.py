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

    def test_retriever_profile_is_published_only_after_api_readiness(self):
        script = self.read("cloud_launch_retriever.sh")
        self.assertIn('RETRIEVER_READY_TIMEOUT=', script)
        self.assertIn('SERVER_PID=$!', script)
        self.assertIn('scripts/cloud_check_retriever.py', script)
        self.assertIn('MARKER_TMP=', script)
        self.assertIn('mv "$MARKER_TMP" "$PROFILE_MARKER"', script)
        self.assertIn('wait "$SERVER_PID"', script)
        self.assertLess(
            script.index('scripts/cloud_check_retriever.py'),
            script.index('mv "$MARKER_TMP" "$PROFILE_MARKER"'),
        )

    def test_smoke_run_calls_smoke_preflight(self):
        script = self.read("cloud_train_grpo_smoke.sh")
        self.assertIn('cloud_preflight.py --profile smoke', script)
        self.assertIn('SEARCH_ENV="${SEARCH_ENV:-Search-R1}"', script)

    def test_smoke_run_writes_success_attestation_after_training(self):
        script = self.read("cloud_train_grpo_smoke.sh")
        self.assertIn('artifacts/retriever_profile.txt', script)
        self.assertIn('profile=smoke', script)
        self.assertIn('artifacts/smoke_passed.txt', script)
        self.assertIn('status=passed', script)
        self.assertIn('kill -0 "$retriever_pid"', script)
        self.assertIn('index_file=$REPO_ROOT/data/smoke_retriever/e5_Flat.index', script)
        self.assertGreater(
            script.index('status=passed'),
            script.index('-m verl.trainer.main_ppo'),
        )

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
        self.assertIn('artifacts/smoke_passed.txt', script)
        self.assertIn('A successful smoke attestation is required', script)
        self.assertIn('artifacts/full_completed.txt', script)
        self.assertIn('kill -0 "$retriever_pid"', script)
        self.assertIn('index_file=$REPO_ROOT/data/wiki18/e5_Flat.index', script)
        self.assertIn('corpus_file=$REPO_ROOT/data/wiki18/wiki-18.jsonl', script)

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

    def test_full_downloads_use_pinned_hugging_face_integrity(self):
        script = self.read("cloud_prepare_data_and_index.sh")
        for expected in (
            "42949672960",
            "21609402413",
            "5123307260",
            "a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023",
            "b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11",
            "7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db",
        ):
            self.assertIn(expected, script)
        self.assertIn('sha256sum "$file"', script)
        self.assertIn('RETRIEVER_ENV="${RETRIEVER_ENV:-Search-R1-retriever}"', script)
        self.assertIn('faiss.read_index', script)
        self.assertIn('downloads.sha256', script)

    def test_full_corpus_decompression_is_atomic(self):
        script = self.read("cloud_prepare_data_and_index.sh")
        self.assertIn('CORPUS_TMP=', script)
        self.assertIn('gzip -cd "$CORPUS_GZ" > "$CORPUS_TMP"', script)
        self.assertIn('mv "$CORPUS_TMP" "$CORPUS_FILE"', script)

    def test_evidence_script_records_reproduction_metadata(self):
        script = self.read("cloud_collect_evidence.sh")
        self.assertNotIn("git remote -v", script)
        self.assertNotIn('for log_file in "$REPO_ROOT"/*.log', script)
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
            'verl_checkpoints',
            'conda run -n "$SEARCH_ENV" python -m pip freeze',
            'conda run -n "$RETRIEVER_ENV" python -m pip freeze',
            'artifacts/smoke_passed.txt',
            'artifacts/full_completed.txt',
            'RUN_STATUS=not_completed',
            'EXPECTED_LOG=',
        ):
            self.assertIn(expected, script)
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("artifacts/", gitignore)

    def test_beginner_docs_use_current_commands(self):
        paths = (
            ROOT / "docs" / "beginner_cloud_reproduction_zh.md",
            ROOT / "docs" / "rental_reproduction_runbook.md",
        )
        for path in paths:
            document = path.read_text(encoding="utf-8")
            for expected in (
                "https://github.com/heba324/Search-R1",
                "Search-R1-retriever",
                "cloud_preflight.py --profile smoke",
                "cloud_prepare_smoke_assets.sh",
                "ASSET_PROFILE=full bash scripts/cloud_launch_retriever.sh",
                "CONFIRM_FULL_RUN=YES bash scripts/cloud_train_grpo_full.sh",
                "cloud_collect_evidence.sh",
                "smoke_passed.txt",
                "profile=full",
                "100 GiB",
            ):
                self.assertIn(expected, document, path.name)
            for obsolete in (
                "conda activate searchr1",
                "conda activate retriever",
                "Search-R1-reproduce",
                "你的用户名",
            ):
                self.assertNotIn(obsolete, document, path.name)

        chinese = paths[0].read_text(encoding="utf-8")
        english = paths[1].read_text(encoding="utf-8")
        self.assertIn("FAISS 与训练共享", chinese)
        self.assertIn("FAISS retriever and training share", english)

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
