import unittest

from scripts.cloud_check_retriever import validate_response


class RetrieverResponseTests(unittest.TestCase):
    def test_accepts_expected_response(self):
        payload = {
            "result": [
                [
                    {
                        "document": {"contents": '"Hamlet"\nText'},
                        "score": 1.0,
                    }
                ]
            ]
        }
        self.assertEqual(validate_response(payload, expected_topk=1)[0]["score"], 1.0)

    def test_rejects_missing_result(self):
        with self.assertRaisesRegex(ValueError, "result"):
            validate_response({}, expected_topk=1)

    def test_rejects_document_without_contents(self):
        payload = {"result": [[{"document": {}, "score": 1.0}]]}
        with self.assertRaisesRegex(ValueError, "contents"):
            validate_response(payload, expected_topk=1)

    def test_rejects_too_few_results(self):
        payload = {
            "result": [
                [{"document": {"contents": '"Hamlet"\nText'}, "score": 1.0}]
            ]
        }
        with self.assertRaisesRegex(ValueError, "expected 3"):
            validate_response(payload, expected_topk=3)

    def test_rejects_boolean_score(self):
        payload = {
            "result": [
                [{"document": {"contents": '"Hamlet"\nText'}, "score": True}]
            ]
        }
        with self.assertRaisesRegex(ValueError, "numeric score"):
            validate_response(payload, expected_topk=1)

    def test_rejects_non_finite_score(self):
        payload = {
            "result": [
                [{"document": {"contents": '"Hamlet"\nText'}, "score": float("nan")}]
            ]
        }
        with self.assertRaisesRegex(ValueError, "finite numeric score"):
            validate_response(payload, expected_topk=1)


if __name__ == "__main__":
    unittest.main()
