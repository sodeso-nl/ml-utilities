import numpy as _np
import pandas as _pd
import tensorflow as _tf
import re as _re
import so_ml_tools as _soml

from typing import Union as _Union

########################################################################################################################
# Word level functions
########################################################################################################################

# Based on TextVectorization
PUNCTUATION = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'
LOWER_AND_STRIP_PUNCTUATION = "lower_and_strip_punctuation"
STRIP_PUNCTUATION = "strip_punctuation"
LOWER = "lower"


def count_unique_words(corpus: _Union[list[str], _tf.Tensor, _np.ndarray, _pd.DataFrame, _pd.Series], standardize="lower_and_strip_punctuation") -> int:
    if not isinstance(corpus, _np.ndarray):
        corpus = _soml.util.types.to_numpy(corpus)

    regex = _re.compile(PUNCTUATION)

    unique = set()
    for line in corpus:
        if isinstance(line, (bytes, bytearray)):
            line = line.decode(encoding='utf-8')

        if standardize in (LOWER, LOWER_AND_STRIP_PUNCTUATION):
            line = line.lower()
        if standardize in (STRIP_PUNCTUATION, LOWER_AND_STRIP_PUNCTUATION):
            line = regex.sub('', line)
        unique = unique.union(set(line.split()))
    return len(unique)


def word_count_for_sentences(corpus: _Union[list[str], _tf.Tensor, _np.ndarray, _pd.DataFrame, _pd.Series]):
    if not isinstance(corpus, _np.ndarray):
        corpus = _soml.util.types.to_numpy(corpus)

    word_count_per_sentence = __count_words_for_each_sentence(corpus)
    word_count_for_90_percentile = int(_np.percentile(a=word_count_per_sentence, q=90))
    word_count_for_95_percentile = int(_np.percentile(a=word_count_per_sentence, q=95))
    word_count_for_97_percentile = int(_np.percentile(a=word_count_per_sentence, q=97))
    word_count_for_99_percentile = int(_np.percentile(a=word_count_per_sentence, q=99))
    word_count_for_max = max(word_count_per_sentence)
    word_count_for_average = round(sum(word_count_per_sentence) / len(corpus))

    print('Word count for x-percentile sentences:')
    print(f' 90%: {word_count_for_90_percentile}')
    print(f' 95%: {word_count_for_95_percentile}')
    print(f' 97%: {word_count_for_97_percentile}')
    print(f' 99%: {word_count_for_99_percentile}')
    print(f'100%: {word_count_for_max}\n')
    print('Average word count for sentences:')
    print(f'Average: {word_count_for_average}')

########################################################################################################################
# Character level functions
########################################################################################################################


def count_unique_chars(corpus: _Union[list[str], _tf.Tensor, _np.ndarray, _pd.DataFrame, _pd.Series], standardize="lower_and_strip_punctuation") -> (int, list[chr]):
    regex = _re.compile(PUNCTUATION)

    if not isinstance(corpus, _np.ndarray):
        corpus = _soml.util.types.to_numpy(corpus)

    unique = set()
    for line in corpus:
        if isinstance(line, (bytes, bytearray)):
            line = line.decode(encoding='utf-8')

        if standardize in (LOWER, LOWER_AND_STRIP_PUNCTUATION):
            line = line.lower()
        if standardize in (STRIP_PUNCTUATION, LOWER_AND_STRIP_PUNCTUATION):
            line = regex.sub('', line)
        unique = unique.union(set(list(line)))
    return len(unique), unique


def length_per_sentence(corpus: _Union[list[str], _tf.Tensor, _np.ndarray, _pd.DataFrame, _pd.Series]) -> list[int]:
    """
    Calculates the length for each string value and returns a list containing
    these lengths.

    :param corpus: a list of string values
    :return:
    """
    if not isinstance(corpus, _np.ndarray):
        corpus = _soml.util.types.to_numpy(corpus)

    return [len(sentence) for sentence in corpus]


def calculate_q_precentile_character_lengths_for_all_sentence(lines: _Union[list[str], _tf.Tensor, _np.ndarray, _pd.DataFrame, _pd.Series], q=95) -> int:
    """
    Calculates the q-percentile based on the lengths of the strings.

    :param lines: a list of string values
    :param q: q-percentile (default 95%)
    :return: the percentile
    """
    return int(_np.percentile(a=length_per_sentence(lines), q=q))


def calculate_average_character_length_for_all_sentences(corpus: _Union[list[str], _tf.Tensor, _np.ndarray, _pd.DataFrame, _pd.Series]) -> int:
    """
    Returns the average length of the strings.

    :param corpus: a list of string values
    :return: the average sentence length
    """
    if not isinstance(corpus, _np.ndarray):
        corpus = _soml.util.types.to_numpy(corpus)

    return round(sum(length_per_sentence(corpus)) / len(corpus))


def __count_words_for_each_sentence(corpus: _Union[list[str], _tf.Tensor, _np.ndarray, _pd.DataFrame, _pd.Series]) -> list[int]:
    """
    Calculates the length for each string value and returns a list containing
    these lengths.

    :param corpus: a list of string values
    :return:
    """
    if not isinstance(corpus, _np.ndarray):
        corpus = _soml.util.types.to_numpy(corpus)

    return [len(i.split()) for i in corpus]
