from collections import defaultdict
import numpy as np
import os

class score_calculator(object):

	def __init__(self, chunk_size: int, syscalls_folder_dir: str, snd_folder_names: [str]):
		self.chunk_size = chunk_size
		self.syscalls_folder_dir = syscalls_folder_dir
		self.syscalls_folder_names = snd_folder_names

	def calculate_average_and_save_to_file(self):

		for name in self.syscalls_folder_names:
			if not os.path.exists(self.syscalls_folder_dir + '/{}/N{}/avgscores'.format(name, self.chunk_size)):
				os.makedirs(self.syscalls_folder_dir + '/{}/N{}/avgscores'.format(name, self.chunk_size))

		for name in self.syscalls_folder_names:
			for j in range(1, 4):
				for i in np.arange(2, 12, 3):
					with open('{}/{}/N{}/{}.{}.labels'.format(self.syscalls_folder_dir, name, self.chunk_size, name, j), 'r') as label_file, open(
							'{}/{}/N{}/results/{}.{}.{}.txt'.format(self.syscalls_folder_dir, name, self.chunk_size, name, j, i), 'r') as result_file, open(
							'{}/{}/N{}/avgscores/{}.{}.{}.avgscores'.format(self.syscalls_folder_dir, name, self.chunk_size, name, j, i), 'w') as avg_score_file:

						counts = defaultdict(int)
						sum = defaultdict(float)
						score_list = [float(score) for score in result_file]

						for i, label in enumerate(label_file):
							counts[label.strip()] += 1
							sum[label.strip()] += score_list[i]

						average = dict()
						for key in counts:
							average[key] = sum[key] / counts[key]

						avg_score = {int(k): v for k, v in average.items()}

						for key in avg_score:
							avg_score_file.write(str(avg_score[key]) + '\n')
