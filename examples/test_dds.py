# encoding utf8


import sys

from pycompss.dds import DDS

from dislib.classification import CascadeSVM


def main():
    file_path = sys.argv[1]
    print("Hola!")

    csvm=CascadeSVM(gamma=0.001, c=10000, max_iter=2000, check_convergence=True,
                    verbose=True)

    def files_to_lines(file_pair):
        return list(file_pair[1])

    def lines_to_dataset(lines, n_features, store_sparse=True):
        from tempfile import SpooledTemporaryFile
        from sklearn.datasets import load_svmlight_file
        from dislib.data import Subset, Dataset

        encoded = [line.encode() for line in lines]

        # Creating a tmp file to use load_svmlight_file method should be more
        # efficient than parsing the lines manually
        tmp_file = SpooledTemporaryFile(mode="wb+", max_size=2e8)
        tmp_file.writelines(encoded)
        tmp_file.seek(0)
        x, y = load_svmlight_file(tmp_file, n_features)

        if not store_sparse:
            x = x.toarray()

        subset = Subset(x, y)

        partition = Dataset(n_features)
        partition.append(subset)
        return partition

    # data = load_libsvm_file(file_path, subset_size=20, n_features=780)

    # data = DDS().load_text_file(
    #     file_path, chunk_size=20, in_bytes=False, strip=False)\
    #     .map_partitions(lines_to_dataset, n_features=780)

    data = DDS().load_files_from_dir(file_path).map_and_flatten(files_to_lines)\
        .map_partitions(lines_to_dataset, n_features=780)

    csvm.fit(data)
    print("Finished")


if __name__ == "__main__":
    main()
