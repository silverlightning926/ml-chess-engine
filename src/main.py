import load_dataset

import numpy


def main():
    boards = load_dataset.getData()
    print(numpy.shape(boards))


if __name__ == '__main__':
    main()
