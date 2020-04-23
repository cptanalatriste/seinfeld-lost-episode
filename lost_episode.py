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
