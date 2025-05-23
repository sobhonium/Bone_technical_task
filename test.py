"""
Author: Sobhan
Date: 23-05-2025
Project: Bones Ltd. job technical challenge. 
File: test.py
Description and Usage:
    unit test file. Just run >> python3 test.py
"""

import unittest
from utils import (read_fillers_from_file,
                   gen_auto_sample_conversation,
                   load_conversation)


class TestMyModule(unittest.TestCase):

    def test_reading_words(self):
        # self.assertEqual(read_words_from_file('transcript.txt'), 8)
        result = read_fillers_from_file('filler_words.txt')
        self.assertIsInstance(result, list)
        self.assertGreater(
            len(result), 0, "Expected at least one word in the result.")
        # self.assertEqual(add(-1, 1), 0)

    def test_all_lines_start_with_speaker(self):
        with open("transcript.txt", "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            if line:  # skip blank lines
                self.assertTrue(
                    line.lower().startswith("speaker "),
                    msg=f"Line {i+1} does not start with 'Speaker': {line}"
                )

    def test_gen_auto_conversation(self):
        conversation = gen_auto_sample_conversation()
        self.assertGreater(len(conversation), 0,
                           "Conversation Should Not Be Empty!")
        self.assertNotIn("Error", conversation)

    def test_load_conversation(self):
        conversation = load_conversation()
        self.assertGreater(len(conversation), 0,
                           "Conversation Should Not Be Empty!")
        self.assertNotIn("Error", conversation)


if __name__ == '__main__':
    unittest.main()
