import unittest

import numpy as np

from align import run_fixtures
from benchmark import edits
from verifier import acoustic_features, decision, has_signal, verify


class PipelineChecks(unittest.TestCase):
    def test_transcript_fixtures(self):
        self.assertTrue(all(result["passed"] for result in run_fixtures()))

    def test_edit_matching_uses_reference_index(self):
        reference = ["a", "b", "c"]
        self.assertFalse(edits(reference, ["a", "c"]) & edits(reference, ["a", "b"]))

    def test_abstention_boundaries(self):
        self.assertEqual(decision(.29, .3, .6), "no_flag")
        self.assertEqual(decision(.3, .3, .6), "review")
        self.assertEqual(decision(.6, .3, .6), "flag")

    def test_signal_gate_rejects_silence_and_low_gain(self):
        silent = np.zeros(16000)
        self.assertFalse(has_signal(silent))
        self.assertIsNone(acoustic_features(silent))
        tone = .1 * np.sin(np.arange(16000) * 2 * np.pi * 400 / 16000)
        self.assertTrue(has_signal(tone))
        self.assertFalse(has_signal(tone * .01778))
        self.assertEqual(acoustic_features(tone).shape, (84,))

    def test_scope(self):
        self.assertIsNone(verify(np.zeros(16000), "112:3"))


if __name__ == "__main__":
    unittest.main()
