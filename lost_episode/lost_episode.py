import torch
from torch import nn


class RNN(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()

        # set  variables
        self.linear_in_features = hidden_dim
        self.lstm_hidden_size = hidden_dim
        self.linear_out_features = output_size
        self.lstm_num_layers = n_layers

        # define model layers
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=self.lstm_num_layers, batch_first=True,
                            dropout=dropout)
        self.linear = nn.Linear(in_features=self.linear_in_features, out_features=self.linear_out_features)

    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        batch_size = nn_input.shape[0]

        embeddings = self.embedding.forward(nn_input)
        lstm_output, hidden_state = self.lstm.forward(embeddings, hidden)

        lstm_stacked_output = lstm_output.contiguous().view(-1, self.linear_in_features)
        linear_output = self.linear.forward(lstm_stacked_output)

        reshape_output = linear_output.view(batch_size, -1, self.linear_out_features)
        return reshape_output[:, -1], hidden_state

    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function

        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            hidden = (weight.new(self.lstm_num_layers, batch_size, self.lstm_hidden_size).zero_().cuda(),
                      weight.new(self.lstm_num_layers, batch_size, self.lstm_hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.lstm_num_layers, batch_size, self.lstm_hidden_size).zero_(),
                      weight.new(self.lstm_num_layers, batch_size, self.lstm_hidden_size).zero_())

        return hidden


def forward_and_backpropagation(network, optimiser, criterion, input_batch, target_batch, hidden_state):
    if torch.cuda.is_available():
        input_batch, target_batch = input_batch.cuda(), target_batch.cuda()

    hidden_state = tuple([each.data for each in hidden_state])

    optimiser.zero_grad()

    forward_output, new_hidden_state = network.forward(input_batch, hidden_state)
    loss_batch = criterion(forward_output, target_batch)

    loss_batch.backward()
    optimiser.step()

    return loss_batch.item(), new_hidden_state
