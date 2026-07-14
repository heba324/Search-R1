import importlib.util
import unittest
from pathlib import Path


class CEGRV2RewardTests(unittest.TestCase):
    def test_mixed_group_reward_is_exactly_the_official_em_vector(self):
        from scripts.improvement_v2.cegr_v2_reward import score_group

        trajectories = [
            "<answer>McComb, Mississippi</answer>",
            (
                "<information>Born in McComb, Mississippi.</information>"
                "<answer>McComb</answer>"
            ),
            "<answer>Houston, Texas</answer>",
        ]

        scored = score_group(trajectories, ["McComb, Mississippi"])

        self.assertEqual([item.exact_match for item in scored], [1.0, 0.0, 0.0])
        self.assertEqual([item.total for item in scored], [1.0, 0.0, 0.0])
        self.assertFalse(any(item.fallback_used for item in scored))

    def test_all_zero_em_group_uses_f1_only_to_restore_relative_order(self):
        from scripts.improvement_v2.cegr_v2_reward import score_group

        trajectories = [
            "<answer>McComb</answer>",
            "<answer>Houston, Texas</answer>",
            "<think>I do not know.</think>",
        ]

        scored = score_group(trajectories, ["McComb, Mississippi"])

        self.assertEqual([item.exact_match for item in scored], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(scored[0].total, 2.0 / 3.0)
        self.assertEqual(scored[1].total, 0.0)
        self.assertEqual(scored[2].total, 0.0)
        self.assertTrue(all(item.fallback_used for item in scored))

    def test_answer_parser_matches_case_sensitive_environment_tags(self):
        from scripts.improvement_v2.cegr_v2_reward import score_group

        uppercase = score_group(
            ["<ANSWER>McComb, Mississippi</ANSWER>"], ["McComb, Mississippi"]
        )[0]
        lowercase = score_group(
            ["<answer>McComb, Mississippi</answer>"], ["McComb, Mississippi"]
        )[0]

        self.assertEqual(uppercase.exact_match, 0.0)
        self.assertEqual(lowercase.exact_match, 1.0)

    def test_answer_score_matches_official_full_sequence_reward(self):
        from scripts.improvement_v2.cegr_v2_reward import score_group

        source = Path(__file__).resolve().parents[1] / "verl/utils/reward_score/qa_em.py"
        spec = importlib.util.spec_from_file_location("official_qa_em", source)
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        prompt_example = "Example: <answer> Beijing </answer>. Question: birthplace?"
        targets = {"target": ["McComb, Mississippi"]}
        for response in (
            "<answer>McComb, Mississippi</answer>",
            "<ANSWER>McComb, Mississippi</ANSWER>",
            "<answer>Houston</answer>",
            "missing tags",
        ):
            with self.subTest(response=response):
                official_score = official.compute_score_em(
                    prompt_example + response, targets
                )
                v2_score = score_group([response], targets["target"])[0].exact_match
                self.assertEqual(v2_score, official_score)

    def test_search_count_and_response_length_do_not_change_answer_reward(self):
        from scripts.improvement_v2.cegr_v2_reward import score_group

        long_multihop = (
            "<think>First identify the singer.</think>"
            "<search>Curious fragrance singer</search>"
            "<information>Curious is a fragrance by Britney Spears.</information>"
            "<think>Now identify her birthplace.</think>"
            "<search>Britney Spears birthplace</search>"
            "<information>Britney Spears was born in McComb, Mississippi.</information>"
            "<answer>McComb, Mississippi</answer>"
        )
        short = "<answer>McComb, Mississippi</answer>"

        scored = score_group([long_multihop, short], ["McComb, Mississippi"])

        self.assertEqual([item.total for item in scored], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
