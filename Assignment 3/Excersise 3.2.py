from data_preprocessor import snd_ns_data_prepocessor as data_preprocessor
import subprocess
from sys import platform


def main():
    data_prep = data_preprocessor(chunk_size=15)
    data_prep.preprocess_and_save_snd_data()

    if platform == "linux" or platform == "linux2":
        subprocess.call('train_test_negative_selection_store_scores.sh')
    elif platform == "darwin":
    # MAC
        raise NotImplementedError("negsel2 MAC OS scripting not yet supported.")
    elif platform == "win32":
    # Windows
        subprocess.call('train_test_negative_selection_store_scores.bat', shell=True)


if __name__ == '__main__':
    main()
