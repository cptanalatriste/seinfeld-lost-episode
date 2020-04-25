import unittest

import torch

import lost_episode
import problem_unittests
from lost_episode import RNN


class TestLostEpisode(unittest.TestCase):

    def setUp(self):
        self.train_on_gpu = torch.cuda.is_available()

    def test_rnn_structure(self):
        problem_unittests.test_rnn(RNN=lost_episode.RNN, train_on_gpu=self.train_on_gpu)

    def test_forward_and_backpropagation(self):
        problem_unittests.test_forward_back_prop(RNN=lost_episode.RNN,
                                                 forward_back_prop=lost_episode.forward_and_backpropagation,
                                                 train_on_gpu=self.train_on_gpu)

    @unittest.skip
    def test_transfer_model_parameters(self):
        model_file = 'test_rnn.pt'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        original_model = torch.load(model_file, map_location=torch.device(device))

        print("original_model", original_model)

        vocab_size = 21388
        output_size = vocab_size
        embedding_dim = 100
        hidden_dim = 64
        n_layers = 2
        test_rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
        print("test_rnn", test_rnn)

        self.assertTrue(
            compare_model_parameters(original_model.state_dict(), test_rnn.state_dict()))


def compare_model_parameters(parameters, more_parameters):
    """
    Taken from: https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/5
    :param parameters:
    :param more_parameters:
    :return:
    """
    models_differ = 0
    for key_item_1, key_item_2 in zip(parameters.items(), more_parameters.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
                return False
    if models_differ == 0:
        return True
