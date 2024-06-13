import load_dataset
import build_model
import train_model


def main():

    data = load_dataset.getData()
    print(data)
    model = build_model.build_model()
    model = train_model.fit_model(model, data)
    train_model.save_model(model, 'model.keras')


if __name__ == '__main__':
    main()
