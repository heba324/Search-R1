import unittest

from scripts.cloud_preflight import HardwareInfo, assess_hardware, parse_gpu_memory


class CloudPreflightTests(unittest.TestCase):
    def test_parse_gpu_memory(self):
        self.assertEqual(parse_gpu_memory("81920\n81920\n"), (81920, 81920))

    def test_smoke_accepts_one_a100_80gb(self):
        info = HardwareInfo((81920,), 128, 300)
        self.assertEqual(assess_hardware("smoke", info), [])

    def test_smoke_rejects_small_gpu(self):
        info = HardwareInfo((24576,), 128, 300)
        errors = "\n".join(assess_hardware("smoke", info))
        self.assertIn("at least 75 GiB VRAM", errors)

    def test_full_requires_eight_40gb_gpus(self):
        info = HardwareInfo((40960,) * 4, 256, 800)
        errors = "\n".join(assess_hardware("full", info))
        self.assertIn("at least 8 NVIDIA GPUs", errors)

    def test_full_checks_ram_and_disk(self):
        info = HardwareInfo((40960,) * 8, 100, 400)
        errors = "\n".join(assess_hardware("full", info))
        self.assertIn("128 GiB RAM", errors)
        self.assertIn("500 GiB free disk", errors)


if __name__ == "__main__":
    unittest.main()
