"""Meaningful Task 2 checks, using validation references and numeric boundaries."""

import csv
import re
import sys
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))

from normalize import normalize
from metrics import canon


class NormalizeTests(unittest.TestCase):
    def test_validation_readings(self):
        with (KIT / "data" / "val_items.csv").open(newline="", encoding="utf-8") as stream:
            items = list(csv.DictReader(stream))
        self.assertEqual(len(items), 24)
        for item in items:
            with self.subTest(item=item["id"]):
                result = normalize(item["text"])
                self.assertFalse(re.search(r"[0-9०-९₹%]", result))
                if item["category"] in ("hindi", "long", "hinglish"):
                    self.assertEqual(result, item["text"])
                else:
                    accepted = [canon(reading) for reading in item["spoken"].split("||")]
                    self.assertIn(canon(result), accepted)

    def test_money_keeps_fraction_and_magnitude(self):
        self.assertEqual(normalize("₹2,340.50"), "दो हज़ार तीन सौ चालीस रुपये पचास पैसे")
        self.assertEqual(normalize("₹2.5 लाख"), "दो दशमलव पाँच लाख रुपये")
        self.assertEqual(normalize("₹0.05"), "शून्य रुपये पाँच पैसे")
        self.assertEqual(normalize("₹1.01"), "एक रुपया एक पैसा")
        self.assertEqual(normalize("₹1.005"), "एक दशमलव शून्य शून्य पाँच रुपये")

    def test_dates_zero_padding_and_percentages(self):
        self.assertEqual(normalize("०५/०६/२०२६"), "पाँच जून दो हज़ार छब्बीस")
        self.assertEqual(normalize("0051"), "शून्य शून्य पाँच एक")
        self.assertEqual(normalize("92%"), "बानवे प्रतिशत")
        self.assertEqual(normalize("2.05%"), "दो दशमलव शून्य पाँच प्रतिशत")

    def test_initialisms_do_not_rewrite_english_words(self):
        self.assertEqual(normalize("RBI ATM GST office internet shopping"),
                         "आर बी आई ए टी एम जी एस टी office internet shopping")
        self.assertEqual(normalize("R.B.I. और slow"), "आर बी आई और slow")

    def test_normalization_is_idempotent(self):
        text = "RBI ने 05/06/2026 को ₹2,340.50 जमा किए, internet slow था।"
        self.assertEqual(normalize(normalize(text)), normalize(text))


if __name__ == "__main__":
    unittest.main()
