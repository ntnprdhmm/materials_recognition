import unittest

from env_functions import get_env

class TestEnv(unittest.TestCase):

    def test_existing_correct_file(self):
        """ Test that the function can parse a valid env file
        """
        env = get_env("tests/tests_assets/valid_env.env")
        self.assertEqual(env["YOLO"], "swagg")
        self.assertEqual(env["N"], 42)
        self.assertEqual(env["PI"], 3.14)
        self.assertEqual(env["FOO"], "bar")

    def test_existing_uncorrect_file(self):
        """ Test that the function fill the dic only with valid lines
        """
        env = get_env("tests/tests_assets/not_valid_env.env")
        self.assertEqual(env["YOLO"], "swagg")
        self.assertEqual(env["FOO"], "bar")

    def test_unexisting_file(self):
        """ Test that the function return default values when no env file
        """
        with self.assertRaises(FileNotFoundError):
            get_env("foo/bar.env")

if __name__ == "__main__":
    unittest.main()
