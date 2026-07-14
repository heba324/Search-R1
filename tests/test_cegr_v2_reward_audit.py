import unittest


class CEGRV2RewardAuditTests(unittest.TestCase):
    def test_audit_proves_mixed_groups_match_em_and_all_zero_groups_gain_variance(self):
        from scripts.improvement_v2.audit_reward_safety import audit_groups

        groups = [
            [
                {
                    "trajectory": "<answer>McComb, Mississippi</answer>",
                    "golden_answers": ["McComb, Mississippi"],
                },
                {
                    "trajectory": "<answer>McComb</answer>",
                    "golden_answers": ["McComb, Mississippi"],
                },
            ],
            [
                {
                    "trajectory": "<answer>McComb</answer>",
                    "golden_answers": ["McComb, Mississippi"],
                },
                {
                    "trajectory": "<answer>Houston</answer>",
                    "golden_answers": ["McComb, Mississippi"],
                },
            ],
        ]

        report = audit_groups(groups, expected_group_size=2)

        self.assertEqual(report["mixed_or_all_correct_groups"], 1)
        self.assertEqual(report["mixed_group_reward_mismatches"], 0)
        self.assertEqual(report["all_zero_groups"], 1)
        self.assertEqual(report["informative_fallback_groups"], 1)
        self.assertTrue(report["safety_invariant_passed"])


if __name__ == "__main__":
    unittest.main()
