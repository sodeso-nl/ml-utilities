import io
import os


def embeddings(directory: str, text_vectorizer, embedding_layer) -> None:
    """
    Exports the embedding weights and the vocabulary of the text vectorizer.

    Use in conjunction with:

    https://projector.tensorflow.org

    :param directory: The target directory to save the weights and vocabulary to.
    :param text_vectorizer: A TextVectorization layer
    :param embedding_layer: An Embedding layer
    """
    # Check if target directory exists, if not, create it
    os.makedirs(directory, exist_ok=True)

    out_v = io.open(os.path.join(directory, 'vectors.tsv'), 'w', encoding='utf-8')
    out_m = io.open(os.path.join(directory, 'metadata.tsv'), 'w', encoding='utf-8')

    vocabulary = text_vectorizer.get_vocabulary()
    embed_weights = embedding_layer.get_weights()[0]

    for index, word in enumerate(vocabulary):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = embed_weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()