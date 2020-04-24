import unittest

import torch

from helper import preprocess_and_save_data
from lost_episode.data_utils import create_int_to_word_map, create_word_to_int_map, create_punctuation_map, \
    create_batch_dataloader
from problem_unittests import test_create_lookup_tables, test_tokenize


class TestDataUtils(unittest.TestCase):

    def test_lookup_tables(self):
        test_create_lookup_tables(create_lookup_tables=create_maps)

    def test_create_punctuation_map(self):
        test_tokenize(token_lookup=create_punctuation_map)

    def test_preprocess_and_save(self):
        preprocess_and_save_data(dataset_path="../data/Seinfeld_Scripts.txt",
                                 token_lookup=create_punctuation_map,
                                 create_lookup_tables=create_maps)

    def test_batch_dataloader(self):
        test_dictionary = [word_index for word_index in range(1, 8)]
        test_lenght = 4
        batch_size = 1
        test_dataloader = create_batch_dataloader(words_as_ints=test_dictionary,
                                                  sequence_lenght=test_lenght,
                                                  batch_size=batch_size)
        data_iterator = iter(test_dataloader)
        first_batch_features, first_batch_target = next(data_iterator)
        self.assertEqual(first_batch_features.shape, (batch_size, test_lenght))
        self.assertEqual(first_batch_target.shape, (1,))

        self.assertTrue(torch.all(first_batch_features.eq(torch.LongTensor([[1, 2, 3, 4]]))))
        self.assertEqual(5, first_batch_target.item())

        second_batch_features, second_batch_target = next(data_iterator)
        self.assertTrue(torch.all(second_batch_features.eq(torch.LongTensor([[2, 3, 4, 5]]))))
        self.assertEqual(6, second_batch_target.item())


def create_maps(text):
    return create_word_to_int_map(text), create_int_to_word_map(text)
