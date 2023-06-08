import numpy as np
from utilities import create_unique_word_dict, load_data, create_word_lists
from keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


window = 2
embed_size = 2  # best practice is for embedding size to be 100, 200, or 300


def create_model_X_y(word_lists, unique_word_dict, n_words):
    X = []
    y = []

    for _, word_list in enumerate(word_lists):
        main_word_index = unique_word_dict.get(word_list[0])
        context_word_index = unique_word_dict.get(word_list[1])

        X_row = np.zeros(n_words)
        Y_row = np.zeros(n_words)

        X_row[main_word_index] = 1
        Y_row[context_word_index] = 1

        X.append(X_row)
        y.append(Y_row)

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


def build_model(X, y):
    inpt = tf.keras.Input(shape=(X.shape[1],))
    x = Dense(units=embed_size, activation='linear')(inpt)
    x = Dense(units=y.shape[1], activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    model.compile(loss ='categorical_crossentropy',
                  optimizer ='adam')
    model.fit(x=X, 
              y=y,
              batch_size=256,
              epochs=1000
              )

    return model


def extract_embeddings(unique_word_dict, words, model):
    weights = model.get_weights()[0]

    embedding_dict = {}
    for word in words: 
        embedding_dict.update({
            word: weights[unique_word_dict.get(word)]
            })
        
    return embedding_dict


def word2vec_embeddings(filename: str, text_col: str):
    samples = load_data(filename=filename,
                        col=text_col)
    word_lists, all_text = create_word_lists(samples=samples,
                                             window=window)
    
    unique_word_dict = create_unique_word_dict(samples)
    n_words = len(unique_word_dict)
    words = list(unique_word_dict.keys())
    
    X, y = create_model_X_y(word_lists=word_lists,
                            unique_word_dict=unique_word_dict,
                            n_words=n_words)
    model = build_model(X, y)
    embeddings = extract_embeddings(unique_word_dict=unique_word_dict,
                                    words=words,
                                    model=model)
    
    return embeddings


if __name__ == '__main__':
    embeddings = word2vec_embeddings(filename='sample.csv', text_col='text')