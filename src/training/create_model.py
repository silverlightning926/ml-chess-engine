from src.training._load_dataset import get_data
from src.training._build_model import build_model
from src.training._train_model import fit_model, save_model


def main():
    data = get_data()
    print(data)
    model = build_model()
    model = fit_model(model, data, verbose=1)
    save_model(model, 'models/model.keras')


if __name__ == '__main__':
    main()
