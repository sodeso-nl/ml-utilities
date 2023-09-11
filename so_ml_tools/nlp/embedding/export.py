import io as _io
import os as _os


def embeddings(folder: str, text_vectorizer, embedding_weights = None, embedding_layer = None, model = None, embedding_layer_name = None) -> None:
    """
    Exports the embedding weights and the vocabulary of the text vectorizer.

    Use in conjunction with:

    https://projector.tensorflow.org

    :param folder: The target folder to save the weights and vocabulary to.
    :param text_vectorizer: A TextVectorization layer
    :param embedding_layer: An Embedding layer
    """
    # Check if target directory exists, if not, create it
    _os.makedirs(folder, exist_ok=True)

    out_v = _io.open(_os.path.join(folder, 'vectors.tsv'), 'w', encoding='utf-8')
    out_m = _io.open(_os.path.join(folder, 'metadata.tsv'), 'w', encoding='utf-8')

    if embedding_weights is None and embedding_layer is not None:
        embedding_weights = embedding_layer.get_weights()[0]
    elif embedding_weights and embedding_layer is None and embedding_layer_name is not None and model is not None:
        embedding_weights = model.get_layer(name=embedding_layer_name).get_weights()[0]

    if embedding_weights is None:
        raise 'No embedding weights specified, either specify embedding_weights or embedding_layer or model and embedding_layer_name.'

    vocabulary = text_vectorizer.get_vocabulary()
    embed_weights = embedding_weights.get_weights()[0]

    for index, word in enumerate(vocabulary):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = embed_weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()
