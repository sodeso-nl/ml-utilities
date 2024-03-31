import numpy as _np
import pandas as _pd
import tensorflow as _tf
import re as _re
from typing import Union as _Union

########################################################################################################################
# Word level functions
########################################################################################################################

# Based on TextVectorization
PUNCTUATION = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'
LOWER_AND_STRIP_PUNCTUATION = "lower_and_strip_punctuation"
STRIP_PUNCTUATION = "strip_punctuation"
LOWER = "lower"


def count_unique_words(lines: _Union[list[str], _tf.Tensor], standardize="lower_and_strip_punctuation") -> int:
    if _tf.is_tensor(x=lines):
        lines = lines.numpy()

    regex = _re.compile(PUNCTUATION)

    unique = set()
    for line in lines:
        if isinstance(line, (bytes, bytearray)):
            line = line.decode(encoding='utf-8')

        if standardize in (LOWER, LOWER_AND_STRIP_PUNCTUATION):
            line = line.lower()
        if standardize in (STRIP_PUNCTUATION, LOWER_AND_STRIP_PUNCTUATION):
            line = regex.sub('', line)
        unique = unique.union(set(line.split()))
    return len(unique)


def calculate_q_precentile_word_lengths(lines: _Union[list[str], _tf.Tensor], q=95) -> int:
    """
    Calculates the q-percentile based on the lengths of the strings.

    :param lines: a list of string values
    :param q: q-percentile (default 95%)
    :return: the percentile
    """
    if _tf.is_tensor(x=lines):
        lines = lines.numpy()

    return int(_np.percentile(a=__count_words_for_each_sentence(lines), q=q))


def maximum_number_of_word_per_sentence(lines: _Union[list[str], _tf.Tensor]) -> int:
    """
    Returns the maximum number of words per sentence.


    :param lines: a list of string values
    :return: the max sentence length in words
    """
    if _tf.is_tensor(x=lines):
        lines = lines.numpy()

    return max(__count_words_for_each_sentence(lines))


def average_number_of_word_per_sentence(lines: _Union[list[str], _tf.Tensor, _pd.DataFrame]) -> int:
    """
    Returns the average number of words per sentence.

    :param lines: a list of string values
    :return: the average sentence length in words
    """
    if _tf.is_tensor(x=lines):
        lines = lines.numpy()

    return round(sum(__count_words_for_each_sentence(lines)) / len(lines))


########################################################################################################################
# Character level functions
########################################################################################################################

def count_unique_chars(lines: _Union[list[str], _tf.Tensor], standardize="lower_and_strip_punctuation") -> (int, list[chr]):
    regex = _re.compile(PUNCTUATION)

    if _tf.is_tensor(x=lines):
        lines = lines.numpy()

    unique = set()
    for line in lines:
        if isinstance(line, (bytes, bytearray)):
            line = line.decode(encoding='utf-8')

        if standardize in (LOWER, LOWER_AND_STRIP_PUNCTUATION):
            line = line.lower()
        if standardize in (STRIP_PUNCTUATION, LOWER_AND_STRIP_PUNCTUATION):
            line = regex.sub('', line)
        unique = unique.union(set(list(line)))
    return len(unique), unique


def length_per_sentence(lines: _Union[list[str], _tf.Tensor]) -> list[int]:
    """
    Calculates the length for each string value and returns a list containing
    these lengths.

    :param lines: a list of string values
    :return:
    """
    if _tf.is_tensor(x=lines):
        lines = lines.numpy()

    return [len(sentence) for sentence in lines]


def calculate_q_precentile_character_lengths_for_all_sentence(lines: _Union[list[str], _tf.Tensor], q=95) -> int:
    """
    Calculates the q-percentile based on the lengths of the strings.

    :param lines: a list of string values
    :param q: q-percentile (default 95%)
    :return: the percentile
    """
    return int(_np.percentile(a=length_per_sentence(lines), q=q))


def calculate_average_character_length_for_all_sentences(lines: _Union[list[str], _tf.Tensor]) -> int:
    """
    Returns the average length of the strings.

    :param lines: a list of string values
    :return: the average sentence length
    """
    if _tf.is_tensor(x=lines):
        lines = lines.numpy()

    return round(sum(length_per_sentence(lines)) / len(lines))


def __count_words_for_each_sentence(lines: _Union[list[str], _tf.Tensor, _pd.Series]) -> list[int]:
    """
    Calculates the length for each string value and returns a list containing
    these lengths.

    :param lines: a list of string values
    :return:
    """
    if isinstance(lines, _pd.DataFrame):
        raise TypeError("lines is a Pandas DataFrame, select a series to calculate the length")
    if isinstance(lines, _pd.Series):
        lines = lines.numpy()
    if _tf.is_tensor(x=lines):
        lines = lines.numpy()

    return [len(i.split()) for i in lines]
