import _load_dataset
import _build_model
import _train_model


def main():
    data = _load_dataset.get_data()
    model = _build_model.build_model()
    model = _train_model.fit_model(model, data)
    _train_model.save_model(model, 'models/model.keras')


if __name__ == '__main__':
    main()
