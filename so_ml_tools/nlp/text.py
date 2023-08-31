import numpy as _np
import re as _re

########################################################################################################################
# Word level functions
########################################################################################################################

# Based on TextVectorization
PUNCTUATION = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'
LOWER_AND_STRIP_PUNCTUATION = "lower_and_strip_punctuation"
STRIP_PUNCTUATION = "strip_punctuation"
LOWER = "lower"


def count_unique_words(lines: list[str], standardize = "lower_and_strip_punctuation") -> int:
    regex = _re.compile(PUNCTUATION)

    unique = set()
    for line in lines:
        if standardize in (LOWER, LOWER_AND_STRIP_PUNCTUATION):
            line = line.lower()
        if standardize in (STRIP_PUNCTUATION, LOWER_AND_STRIP_PUNCTUATION):
            line = regex.sub('', line)
        unique = unique.union(set(line.split()))
    return len(unique)


def count_words_for_each_sentence(lines: list[str]) -> list[int]:
    """
    Calculates the length for each string value and returns a list containing
    these lengths.

    :param lines: a list of string values
    :return:
    """
    return [len(i.split()) for i in lines]


def calculate_q_precentile_word_lengths(lines: list[str], q=95) -> int:
    """
    Calculates the q-percentile based on the lengths of the strings.

    :param lines: a list of string values
    :param q: q-percentile (default 95%)
    :return: the percentile
    """
    return int(_np.percentile(a=count_words_for_each_sentence(lines), q=q))


def average_number_of_word_per_sentence(lines: list[str]) -> int:
    """
    Returns the average length of the strings.

    :param lines: a list of string values
    :return: the average sentence length
    """
    return round(sum(count_words_for_each_sentence(lines)) / len(lines))


########################################################################################################################
# Character level functions
########################################################################################################################

def count_unique_chars(lines: list[str], standardize = "lower_and_strip_punctuation") -> (int, list[chr]):
    regex = _re.compile(PUNCTUATION)

    unique = set()
    for line in lines:
        if standardize in (LOWER, LOWER_AND_STRIP_PUNCTUATION):
            line = line.lower()
        if standardize in (STRIP_PUNCTUATION, LOWER_AND_STRIP_PUNCTUATION):
            line = regex.sub('', line)
        unique = unique.union(set(list(line)))
    return len(unique), unique


def length_per_sentence(lines: list[str]) -> list[int]:
    """
    Calculates the length for each string value and returns a list containing
    these lengths.

    :param lines: a list of string values
    :return:
    """
    return [len(sentence) for sentence in lines]


def calculate_q_precentile_character_lengths_for_all_sentence(lines: list[str], q=95) -> int:
    """
    Calculates the q-percentile based on the lengths of the strings.

    :param lines: a list of string values
    :param q: q-percentile (default 95%)
    :return: the percentile
    """
    return int(_np.percentile(a=length_per_sentence(lines), q=q))


def calculate_average_character_length_for_all_sentences(lines: list[str]) -> int:
    """
    Returns the average length of the strings.

    :param lines: a list of string values
    :return: the average sentence length
    """
    return round(sum(length_per_sentence(lines)) / len(lines))