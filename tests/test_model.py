import unittest

from genesis.model import ModelUnavailable, validate_configs


class ValidateConfigsTests(unittest.TestCase):
    def test_accepts_only_unique_allowlisted_parameters(self):
        result = validate_configs([
            {"block_size": 256, "num_warps": 4, "rationale": "coalesced reads"},
            {"block_size": 256, "num_warps": 4, "rationale": "duplicate"},
            {"block_size": 7, "num_warps": 4},
            {"block_size": 512, "num_warps": 8},
        ], attempts=3)
        self.assertEqual([(x["block_size"], x["num_warps"]) for x in result], [(256, 4), (512, 8)])

    def test_caps_result_to_attempt_budget(self):
        source = [{"block_size": block, "num_warps": 4} for block in (128, 256, 512, 1024)]
        self.assertEqual(len(validate_configs(source, attempts=2)), 2)

    def test_rejects_empty_or_malformed_result(self):
        with self.assertRaises(ModelUnavailable):
            validate_configs([{"block_size": 13, "num_warps": 3}], attempts=1)
        with self.assertRaises(ModelUnavailable):
            validate_configs({"block_size": 128}, attempts=1)


if __name__ == "__main__":
    unittest.main()
