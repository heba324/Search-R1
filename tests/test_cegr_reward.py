import unittest


class CEGRRewardTests(unittest.TestCase):
    def test_token_f1_gives_partial_credit_without_relaxing_exact_match(self):
        from scripts.improvement.cegr_reward import score_trajectory

        result = score_trajectory(
            "<answer>McComb</answer>",
            ["McComb, Mississippi"],
            step=1,
            total_steps=120,
        )

        self.assertEqual(result.exact_match, 0.0)
        self.assertAlmostEqual(result.token_f1, 2.0 / 3.0)
        self.assertGreater(result.total, 0.0)

    def test_correct_answer_always_outranks_evidence_only_trajectory(self):
        from scripts.improvement.cegr_reward import score_trajectory

        correct = score_trajectory(
            "<answer>McComb, Mississippi</answer>",
            ["McComb, Mississippi"],
            step=1,
            total_steps=120,
        )
        evidence_only = score_trajectory(
            "<search>Britney Spears birthplace</search>"
            "<information>Britney Spears was born in McComb, Mississippi.</information>"
            "<answer>Houston, Texas</answer>",
            ["McComb, Mississippi"],
            step=1,
            total_steps=120,
        )

        self.assertGreater(correct.total, evidence_only.total)

    def test_schedule_moves_weight_from_exploration_to_exactness(self):
        from scripts.improvement.cegr_reward import reward_weights

        early = reward_weights(step=1, total_steps=120)
        late = reward_weights(step=120, total_steps=120)

        self.assertLess(early.em_share, late.em_share)
        self.assertGreater(early.evidence_weight, late.evidence_weight)
        self.assertLess(early.behavior_penalty_weight, late.behavior_penalty_weight)

    def test_answer_coverage_requires_informative_gold_alias(self):
        from scripts.improvement.cegr_reward import evidence_answer_coverage

        trajectory = "<information>Yes, the article discusses it.</information>"
        self.assertEqual(evidence_answer_coverage(trajectory, ["yes"]), 0.0)
        self.assertEqual(
            evidence_answer_coverage(
                "<information>Spears was born in McComb, Mississippi.</information>",
                ["McComb, Mississippi"],
            ),
            1.0,
        )

    def test_invalid_duplicate_and_excess_searches_increase_penalty(self):
        from scripts.improvement.cegr_reward import search_behavior_penalty

        clean = search_behavior_penalty(
            "<search>Curious fragrance singer</search>"
            "<search>Britney Spears birthplace</search>"
        )
        noisy = search_behavior_penalty(
            "<search>Britney Spears birthplace</search>"
            "<search>britney   spears birthplace</search>"
            "<search> </search><search>another query</search>"
        )

        self.assertEqual(clean, 0.0)
        self.assertGreater(noisy, clean)
        self.assertLessEqual(noisy, 1.0)


if __name__ == "__main__":
    unittest.main()
