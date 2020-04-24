import unittest

import torch

import lost_episode
import problem_unittests


class TestLostEpisode(unittest.TestCase):

    def setUp(self):
        self.train_on_gpu = torch.cuda.is_available()

    def test_rnn_structure(self):
        problem_unittests.test_rnn(RNN=lost_episode.RNN, train_on_gpu=self.train_on_gpu)

    def test_forward_and_backpropagation(self):
        problem_unittests.test_forward_back_prop(RNN=lost_episode.RNN,
                                                 forward_back_prop=lost_episode.forward_and_backpropagation,
                                                 train_on_gpu=self.train_on_gpu)
