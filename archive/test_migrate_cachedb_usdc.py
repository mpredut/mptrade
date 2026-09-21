import json
import tempfile
import unittest
from pathlib import Path

from verify_tools.migrate_cachedb_usdc import migrate_file


class CachedbUsdcMigrationTest(unittest.TestCase):
    def test_json_migration_is_atomic_backed_up_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(json.dumps({
                "TOTAL": [{"total_value_usdt": 123}],
                "BTCUSDT": {"symbol": "TAOUSDT"},
            }), encoding="utf-8")
            self.assertEqual(migrate_file(path, apply=True), 3)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["TOTAL"][0]["total_value_usdc"], 123)
            self.assertIn("BTCUSDC", data)
            self.assertEqual(data["BTCUSDC"]["symbol"], "TAOUSDC")
            self.assertTrue(Path(str(path) + ".pre_usdc_migration").exists())
            self.assertEqual(migrate_file(path, apply=True), 0)


if __name__ == "__main__":
    unittest.main()
