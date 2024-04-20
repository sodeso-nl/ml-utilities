import numpy as _np
import numpy as np
import pandas as _pd
import tensorflow as _tf
import nltk as _nltk
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


def strip_punctuation(text: _Union[str, list[str], _tf.Tensor, _np.ndarray, _pd.DataFrame, _pd.Series]) -> _Union[str, list[str]]:
    """
    Strip punctuation from input text.

    Usage:
    >>> soml.nlp.text.strip_punctuation(data['text'])
    >>>

    Args:
        text: Text to be stripped

    Returns:
        The stripped text
    """
    if isinstance(text, str):
        converted_text = np.array([text])
    elif not isinstance(text, _np.ndarray):
        converted_text = _soml.util.types.to_numpy(text)
    else:
        converted_text = text

    regex = _re.compile(PUNCTUATION)

    def strip_sentence(sentence):
        if isinstance(sentence, (bytes, bytearray)):
            sentence = sentence.decode(encoding='utf-8')

        return regex.sub('', sentence)

    result = [strip_sentence(sentence) for sentence in converted_text]

    if isinstance(text, str):
        return result[0]

    return result


def nltk_lemmatize(text: _Union[str, list[str], _tf.Tensor, _np.ndarray, _pd.DataFrame, _pd.Series]) -> _Union[str, list[str]]:
    """
    Lemmatize the the given sentence.

    Usage:
    >>> data['text'] = soml.nlp.text.nltk_lemmatize(data['text']))

    Args:
        text: the text to lemmatize.

    Returns:
        The lemmatized sentence
    """
    if isinstance(text, str):
        converted_text = np.array([text])
    elif not isinstance(text, _np.ndarray):
        converted_text = _soml.util.types.to_numpy(text)
    else:
        converted_text = text

    lemmatizer = _nltk.stem.WordNetLemmatizer()

    def lemmatize_sentence(sentence):
        # tokenize the sentence and find the POS tag for each token
        nltk_tagged = _nltk.pos_tag(_nltk.word_tokenize(sentence))
        # tuple of (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], _nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                # if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:
                # else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        return " ".join(lemmatized_sentence)

    return [lemmatize_sentence(sentence) for sentence in converted_text]


def _nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return _nltk.corpus.wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return _nltk.corpus.wordnet.VERB
    elif nltk_tag.startswith('N'):
        return _nltk.corpus.wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return _nltk.corpus.wordnet.ADV
    else:
        return None


def nltk_remove_stopwords_english(text: _Union[str, list[str], _tf.Tensor, _np.ndarray, _pd.DataFrame, _pd.Series]) -> _Union[str, list[str]]:
    """
    Remove stopwords from the given text

    Usage:
    >>> data['text'] = soml.nlp.text.nltk_remove_stopwords_english(data['text']))

    Args:
        text: the text to filter out stopwords.

    Returns:
        The filtered text
    """
    if isinstance(text, str):
        converted_text = np.array([text])
    elif not isinstance(text, _np.ndarray):
        converted_text = _soml.util.types.to_numpy(text)
    else:
        converted_text = text

    stop_words = _nltk.corpus.stopwords.words('english')

    def remove_stopwords(sentence):
        word_tokens = _nltk.word_tokenize(sentence)
        filtered_sentence = [word for word in word_tokens if not word.lower() in stop_words]
        return ' '.join(filtered_sentence)

    return [remove_stopwords(sentence) for sentence in converted_text]


def remove_empty_lines(text: _Union[list[str], _tf.Tensor, _np.ndarray, _pd.DataFrame, _pd.Series]) -> _np.array:
    if not isinstance(text, _np.ndarray):
        converted_text = _soml.util.types.to_numpy(text)
    else:
        converted_text = text

    return converted_text[_np.char.str_len(converted_text.astype(np.str_)) > 0]


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


def text_statistics(corpus: _Union[list[str], _tf.Tensor, _np.ndarray, _pd.DataFrame, _pd.Series]):
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
