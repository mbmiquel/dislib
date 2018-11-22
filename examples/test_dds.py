# encoding: utf-8

from pycompss.dds import DDS
import sys
from dislib.classification import CascadeSVM
from dislib.data import Dataset
from dislib.data.base import _read_libsvm


def main():
    """

    """
    file_path = sys.argv[1]

    csvm = CascadeSVM(cascade_arity=2, max_iter=5, c=1000, gamma=0.01,
                      check_convergence=True)

    print("STARTED")






if __name__ == "__main__":
    main()
