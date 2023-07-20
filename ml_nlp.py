import string
from collections import Counter

import numpy as np
import re

########################################################################################################################
# Word level functions
########################################################################################################################

# Based on TextVectorization
PUNCTUATION = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'
LOWER_AND_STRIP_PUNCTUATION = "lower_and_strip_punctuation"
STRIP_PUNCTUATION = "strip_punctuation"
LOWER = "lower"


def count_unique_words(lines: list[str], standardize = "lower_and_strip_punctuation") -> int:
    regex = re.compile(PUNCTUATION)

    unique = set()
    for line in lines:
        if standardize in (LOWER, LOWER_AND_STRIP_PUNCTUATION):
            line = line.lower()
        if standardize in (STRIP_PUNCTUATION, LOWER_AND_STRIP_PUNCTUATION):
            line = regex.sub('', line)
        unique = unique.union(set(line.split()))
    return len(unique)


def calculate_word_lengths(lines: list[str]) -> list[int]:
    """
    Calculates the length for each string valaue and returns a list containing
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
    return int(np.percentile(a=calculate_word_lengths(lines), q=q))


def calculate_average_word_length(lines: list[str]) -> int:
    """
    Returns the average length of the strings.

    :param lines: a list of string values
    :return: the average sentence length
    """
    return round(sum(calculate_word_lengths(lines)) / len(lines))


########################################################################################################################
# Character level functions
########################################################################################################################

def count_unique_chars(lines: list[str], standardize = "lower_and_strip_punctuation") -> (int, list[chr]):
    regex = re.compile(PUNCTUATION)

    unique = set()
    for line in lines:
        if standardize in (LOWER, LOWER_AND_STRIP_PUNCTUATION):
            line = line.lower()
        if standardize in (STRIP_PUNCTUATION, LOWER_AND_STRIP_PUNCTUATION):
            line = regex.sub('', line)
        unique = unique.union(set(list(line)))
    return len(unique), unique


def calculate_character_lengths(lines: list[str]) -> list[int]:
    """
    Calculates the length for each string valaue and returns a list containing
    these lengths.

    :param lines: a list of string values
    :return:
    """
    return [len(sentence) for sentence in lines]


def calculate_q_precentile_character_lengths(lines: list[str], q=95) -> int:
    """
    Calculates the q-percentile based on the lengths of the strings.

    :param lines: a list of string values
    :param q: q-percentile (default 95%)
    :return: the percentile
    """
    return int(np.percentile(a=calculate_character_lengths(lines), q=q))


def calculate_average_character_length(lines: list[str]) -> int:
    """
    Returns the average length of the strings.

    :param lines: a list of string values
    :return: the average sentence length
    """
    return round(sum(calculate_character_lengths(lines)) / len(lines))