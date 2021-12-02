import tensorflow as tf
import sys
from matplotlib import pyplot
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

MODEL_EXPORT_PATH = '../model.h5'


def get_model():
    model = keras.models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Rescaling(scale=1 / 255.0),
        layers.Conv2D(32, (7, 7), activation='relu', kernel_initializer='he_uniform', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (7, 7), activation='relu', kernel_initializer='he_uniform', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    my_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def image_augmentation(x_train):
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(x_train)
    return datagen


def train_checkpoints():
    checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=MODEL_EXPORT_PATH,
                                                    monitor='val_accuracy',
                                                    mode='max',
                                                    verbose=1,
                                                    save_best_only=True)
    return checkpoint_cb


if __name__ == '__main__':
    dataset = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = dataset
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    TRAIN = False

    if TRAIN:
        model = get_model()
        print(model.summary())
        datagen = image_augmentation(x_train)
        callback_list = [train_checkpoints()]

        history = model.fit(datagen.flow(x_train, y_train, batch_size=128),
                            steps_per_epoch=len(x_train) / 128, epochs=100, validation_data=(x_test, y_test),
                            callbacks=callback_list)

    else:
        from tensorflow.keras.datasets import cifar10
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.models import load_model

        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

        modelTEST = load_model("../model.h5")
        scores = modelTEST.evaluate(X_test, to_categorical(Y_test))
        modelTEST.summary()
        print('Loss: %.3f' % scores[0])
        print('Accuracy: %.3f' % scores[1])