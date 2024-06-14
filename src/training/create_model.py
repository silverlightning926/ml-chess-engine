import training._load_dataset as _load_dataset
import training._build_model as _build_model
import training._train_model as _train_model


def main():

    data = _load_dataset.getData()
    model = _build_model.build_model()
    model = _train_model.fit_model(model, data)
    _train_model.save_model(model, '../../models/model.keras')


if __name__ == '__main__':
    main()
