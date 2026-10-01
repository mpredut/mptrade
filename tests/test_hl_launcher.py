import importlib.util
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SPEC = importlib.util.spec_from_file_location(
    "repo_hl_bot", os.path.join(ROOT, "hyperliquid", "hl_bot.py")
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
state_dir_for = MODULE.state_dir_for


class HLLauncherTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_here = MODULE._HERE
        MODULE._HERE = self.tempdir.name

    def tearDown(self):
        MODULE._HERE = self.original_here
        self.tempdir.cleanup()

    def test_live_state_is_hyperliquid_scoped(self):
        self.assertEqual(state_dir_for(False), self.tempdir.name)

    def test_paper_state_is_created_and_separate_from_live(self):
        paper_dir = state_dir_for(True)
        self.assertEqual(paper_dir, os.path.join(self.tempdir.name, ".paper_state"))
        self.assertTrue(os.path.isdir(paper_dir))
        self.assertNotEqual(paper_dir, state_dir_for(False))


if __name__ == "__main__":
    unittest.main()
