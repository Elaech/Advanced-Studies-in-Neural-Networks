from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb


from tensorflow.keras.preprocessing import sequence

NR_UNIQUE_WORDS = 6000
MAX_REVIEW_LENGTH = 200
EMBED_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 100
MODEL_EXPORT_PATH = "../model.h5"


def get_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=NR_UNIQUE_WORDS, input_length=MAX_REVIEW_LENGTH, output_dim=EMBED_SIZE),
        layers.Dropout(0.4),
        layers.LSTM(80),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    opt = keras.optimizers.RMSprop()
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_save_cp_callback():
    checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=MODEL_EXPORT_PATH,
                                                    monitor='val_accuracy',
                                                    mode='max',
                                                    verbose=1,
                                                    save_best_only=True)


if __name__ == '__main__':
    TRAIN = False
    if TRAIN:
        (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=NR_UNIQUE_WORDS)
        X_train = sequence.pad_sequences(X_train, maxlen=MAX_REVIEW_LENGTH)
        X_test = sequence.pad_sequences(X_test, maxlen=MAX_REVIEW_LENGTH)
        model = get_model()
        callback_list = [get_save_cp_callback()]
        history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                            validation_data=(X_test, Y_test),
                            callbacks=callback_list)

    else:
        from tensorflow.keras.datasets import imdb
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import sequence
        import sys

        # numarul maxim de cuvinte pe care vreti sa le considerati
        nr_cuv_diferite = NR_UNIQUE_WORDS  # ex: 5000
        # dimensiunea maxima a unui review
        dim_max = MAX_REVIEW_LENGTH  # ex: 500

        # Load the dataset
        (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=nr_cuv_diferite)
        model = load_model(MODEL_EXPORT_PATH)
        X_test = sequence.pad_sequences(X_test, maxlen=dim_max)
        scores = model.evaluate(X_test, Y_test)
        model.summary()
        print('Loss: %.3f' % scores[0])
        print('Accuracy: %.3f' % scores[1])

