import unittest


class CEGRV2GroupingTests(unittest.TestCase):
    def test_rollouts_share_uid_by_dataset_split_and_example_after_reordering(self):
        from scripts.improvement_v2.grouping import build_group_uids

        data_sources = ["nq", "hotpotqa", "nq", "hotpotqa"]
        extra_infos = [
            {"split": "train", "index": 7},
            {"split": "train", "index": 3},
            {"split": "train", "index": 7},
            {"split": "train", "index": 3},
        ]

        uids = build_group_uids(data_sources, extra_infos, expected_group_size=2)

        self.assertEqual(uids[0], uids[2])
        self.assertEqual(uids[1], uids[3])
        self.assertNotEqual(uids[0], uids[1])

    def test_grouping_fails_closed_when_any_prompt_has_wrong_rollout_count(self):
        from scripts.improvement_v2.grouping import build_group_uids

        with self.assertRaisesRegex(ValueError, "expected 2"):
            build_group_uids(
                ["nq", "nq", "hotpotqa"],
                [
                    {"split": "train", "index": 7},
                    {"split": "train", "index": 7},
                    {"split": "train", "index": 3},
                ],
                expected_group_size=2,
            )

    def test_batch_scoring_is_group_correct_after_arbitrary_reordering(self):
        from scripts.improvement_v2.cegr_v2_reward import score_batch_by_uid

        uids = ["nq:train:7", "hotpotqa:train:3", "nq:train:7", "hotpotqa:train:3"]
        trajectories = [
            "<answer>McComb, Mississippi</answer>",
            "<answer>McComb</answer>",
            "<answer>McComb</answer>",
            "<answer>Houston</answer>",
        ]
        targets = [["McComb, Mississippi"]] * 4

        eff = score_batch_by_uid(uids, trajectories, targets, mode="eff")
        control = score_batch_by_uid(uids, trajectories, targets, mode="grouped_em")

        self.assertEqual([item.total for item in eff], [1.0, 2.0 / 3.0, 0.0, 0.0])
        self.assertEqual([item.total for item in control], [1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
