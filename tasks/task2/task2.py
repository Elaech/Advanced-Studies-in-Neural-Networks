from tasks.task2 import data_preprocess
from tasks.task2.model import init_model, save_model, load_model
from tasks.task2.model_training import train_model, Params, test_model


def main():
    TRAIN = False

    if TRAIN:
        data = data_preprocess.get_train_data()
        model = init_model(Params.INPUT_SIZE, Params.LABEL_COUNT)
        model = train_model(model, data)
        if not Params.SAVE_BEST:
            save_model(model)
    else:
        model = load_model()
        data = data_preprocess.get_test_data()
        test_model(model, data)


if __name__ == '__main__':
    main()
