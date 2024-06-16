  # Nova Machine Learning Chess Engine
*<div style="color:gray;margin-top:-10px;">By Siddharth Rao</div>*

---

![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=flat&logo=Keras&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?style=flat&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white) ![Dataset](https://img.shields.io/badge/Dataset-Lichess_Chess_Game_Dataset-blue)

---

![Python Lint - autopep8 Workflow](https://github.com/silverlightning926/tensorflow-chess-engine/actions/workflows/python-lint.yaml/badge.svg)

---

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

  - [Summary](#summary)
  - [Current Status](#current-status)
  - [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
- [Linting With Autopep8](#linting-with-autopep8)
- [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
  - [Results](#results)
  - [License](#license)

<!-- /code_chunk_output -->

---

## Summary
This repository contains code for the nova chess engine. The repository contains the code for both the evaluation machine learning model built with TensorFlow as well as the algorithim to find the best move given a position. 

## Current Status
The project is currently in the development phase. The evaluation model has been trained on the Lichess Chess Game Dataset and the model is currently being tested. The code for the move generation algorithm is currently being developed. ⚠️

## Getting Started
To get started with developing this project:

1. Clone the repository to your local machine:
   ```bash
    git clone https://github.com/silverlightning926/tensorflow-chess-engine.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Train the model running the [create_model.py](./src/training/create_model.py) script
    ```bash
    python src.training.create_model.py
    ```
4. Test the model running the [player_vs_model.py.py](./src/play/player_vs_model.py) script
    ```bash
    python src.play.player_vs_model.py
    ```

## Dependencies
The project uses the following dependencies:
- Python 3.12.4
- Tensorflow 2.16.1
- Keras 3.3.3
- Chess 1.10.0
- Kaggle 1.6.14
- NumPy 1.26.4
- Pandas 2.2.2
- Tqdm 4.66.4

# Linting With Autopep8
To ensure consistent code formatting, you can use Autopep8, a Python library that automatically formats your code according to the PEP 8 style guide. To install Autopep8, run the following command:
```bash
pip install autopep8
```

Once installed, you can use Autopep8 to automatically format your code by running the following command:
```bash
autopep8 --in-place --recursive ./src
```

This will recursively format all Python files in the current directory and its subdirectories.

Remember to run Autopep8 regularly to maintain a clean and consistent codebase. This repo contains the Python Lint GitHub Workflow to ensure the repository stays linted.

If you are using VSCode, you can download and the [Autopep8 VSCode Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.autopep8) and add these lines to your `settings.json` to format with Autopep8 automatically as you type and when you save.
```json
"[python]": {
        "editor.formatOnType": true,
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "ms-python.autopep8"
    }
```

# Dataset
The dataset used to train the model is the [Lichess Chess Game Dataset](https://www.kaggle.com/datasnaek/chess) available on Kaggle. The dataset contains over 20 million chess games played on Lichess, a popular online chess platform. The dataset includes information about the moves played in each game, as well as the ratings of the players and the outcome of the game. The dataset is available in a CSV format and can be downloaded from the Kaggle website.

If you are running [create_model.py](./src/training/create_model.py) the script will automatically download the dataset (if you have Kaggle API authentication setup) from Kaggle and save it to the [data](./data/) directory. After processing the data, the data will cache the processed data to the [data](./data/) directory.

## Model Architecture
The model architecture used in this project is built here [_build_model.py](./src/training/_build_model.py)

The input to the model is a group of 8x8x12 matrices, where each matrix represents the board state at a different point in time, they are then grouped together as a single game.

The model consists of a series of convolutional layers, batch normalization layers, GRU layers, and more followed by a series of dense layers. The output of the model is a single value representing the predicted outcome of the game. The model is trained using the mean squared error loss function and the Adam optimizer.

The model then ends add a Dense layer with a single output. This layer outputs a continous number between -1 to 1. 1 signifies a win for white, -1 signifies a win for black, and 0 signifies a draw. The model is trained to predict the outcome of the game based on the board state.

## Results
The engine is currently being trained and tested. The results will be updated here once the model and engine has been evaluated.

## License
This repository is governed under the MIT license. The repository's license can be found here: [LICENSE](./LICENSE).