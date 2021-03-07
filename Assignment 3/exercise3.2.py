from data_preprocessor import snd_ns_data_prepocessor as data_preprocessor
import subprocess
from sys import platform
from score_calculator import score_calculator
from plotter import ROC_AUC_plotter


def main():
    snd_folder_names = ['snd-cert', 'snd-unm']
    syscalls_file_dir = 'negative-selection/syscalls'
    chunk_size = 15
    data_prep = data_preprocessor(chunk_size=chunk_size, snd_folder_names=snd_folder_names, syscalls_file_dir=syscalls_file_dir)
    data_prep.preprocess_and_save_snd_data()

    #Tested on windows! Should work on linux. dont forget to set chunk size and r values in scripts if changed!
    #TODO: support for parameter passing to scripts.
    if platform == "linux" or platform == "linux2":
        subprocess.call('train_test_negative_selection_store_scores.sh')
    elif platform == "darwin":
    # MAC
        raise NotImplementedError("negsel2 MAC OS scripting not yet supported.")
    elif platform == "win32":
    # Windows
        subprocess.call('train_test_negative_selection_store_scores.bat', shell=True)

    score_calc = score_calculator(syscalls_folder_dir=syscalls_file_dir, snd_folder_names=snd_folder_names, chunk_size=chunk_size)
    score_calc.calculate_average_and_save_to_file()

    plotter = ROC_AUC_plotter(syscalls_folder_dir=syscalls_file_dir, snd_folder_names=snd_folder_names, chunk_size=chunk_size)
    plotter.plot_and_save_ROC_AUC()


if __name__ == '__main__':
    main()
