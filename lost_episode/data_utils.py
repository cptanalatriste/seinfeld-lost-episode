import torch
from torch.utils.data import DataLoader, TensorDataset


def create_word_to_int_map(word_list):
    return {word: index for index, word in enumerate(set(word_list))}


def create_int_to_word_map(word_list):
    return {index: word for index, word in enumerate(set(word_list))}


def create_punctuation_map():
    return {'.': "||period||",
            ',': "||comma||",
            '"': "||quotation_mark||",
            ';': "||semicolon||",
            '!': "||exclamation_mark||",
            '?': "||question_mark||",
            '(': "||left_parentheses||",
            ')': "||right_parentheses||",
            '-': "||dash||",
            '\n': "||return||"}


def create_batch_dataloader(words_as_ints, sequence_lenght, batch_size, shuffle=True):
    sequence_start = 0
    feature_tensor = []
    target_tensor = []

    while sequence_start < len(words_as_ints) - sequence_lenght:
        target_index = sequence_start + sequence_lenght

        feature_tensor.append(words_as_ints[sequence_start:target_index])
        target_tensor.append(words_as_ints[target_index])

        sequence_start += 1

    dataset = TensorDataset(torch.LongTensor(feature_tensor), torch.LongTensor(target_tensor))
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
