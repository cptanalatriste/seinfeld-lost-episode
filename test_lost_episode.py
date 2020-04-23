import unittest

from helper import preprocess_and_save_data
from lost_episode import create_int_to_word_map, create_word_to_int_map, create_punctuation_map
from problem_unittests import test_create_lookup_tables, test_tokenize


class TestLostEpisode(unittest.TestCase):

    def test_lookup_tables(self):
        test_create_lookup_tables(create_lookup_tables=create_maps)

    def test_create_punctuation_map(self):
        test_tokenize(token_lookup=create_punctuation_map)

    def test_preprocess_and_save(self):
        preprocess_and_save_data(dataset_path="data/Seinfeld_Scripts.txt",
                                 token_lookup=create_punctuation_map,
                                 create_lookup_tables=create_maps)


def create_maps(text):
    return create_word_to_int_map(text), create_int_to_word_map(text)
