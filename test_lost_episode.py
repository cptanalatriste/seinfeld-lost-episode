import unittest
from lost_episode import create_int_to_word_map, create_word_to_int_map, create_punctuation_map
from problem_unittests import test_create_lookup_tables, test_tokenize


class TestLostEpisode(unittest.TestCase):

    def test_lookup_tables(self):
        test_create_lookup_tables(create_lookup_tables=input_function)

    def test_create_punctuation_map(self):
        test_tokenize(token_lookup=create_punctuation_map)


def input_function(text):
    return create_int_to_word_map(text), create_word_to_int_map(text)
