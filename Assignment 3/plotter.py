from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os

class ROC_AUC_plotter(object):
    def __init__(self, syscalls_folder_dir: str, snd_folder_names: [str], chunk_size: int):
        self.syscalls_folder_dir = syscalls_folder_dir
        self.snd_folder_names = snd_folder_names
        self.chunk_size = chunk_size

    def plot_and_save_ROC_AUC(self,):
        false_positive_rate = dict()
        true_positive_rate = dict()
        roc_auc = dict()

        test_file_name_score = self.syscalls_folder_dir + '/{}/N{}/avgscores/{}.{}.{}.avgscores'
        labels_file_name = self.syscalls_folder_dir + '/{}/{}.{}.labels'
        r_values = np.arange(2, 12, 3)

        for syscalls_folder_name in self.snd_folder_names: #for both folders or a single folder if length syscall folder <2.
            for number in range(1, 4):  # number of syscall files per syscall folder is 3.
                for r in r_values:
                    with open(test_file_name_score.format(syscalls_folder_name, self.chunk_size, syscalls_folder_name, number, r),
                              'r') as test_file, open(
                            labels_file_name.format(syscalls_folder_name, syscalls_folder_name, number), 'r') as labels_file:
                        labels = np.array(labels_file.read().splitlines()).astype(np.float64)
                        prediction = np.array(test_file.read().splitlines()).astype(np.float64)
                        prediction = prediction / float(max(prediction))

                        false_positive_rate[r], true_positive_rate[r], _ = roc_curve(labels, prediction)
                        roc_auc[r] = auc(false_positive_rate[r], true_positive_rate[r])

                plt.figure()
                for r in r_values:
                    plt.plot(false_positive_rate[r], true_positive_rate[r], lw=2,
                             label='ROC r={} (area = {:0.2f})'.format(r, roc_auc[r]))

                plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC curve for {}-{}'.format(syscalls_folder_name, number))
                plt.legend(loc="lower right")
                plt.savefig('{}.{}.png'.format(syscalls_folder_name, number))
