import os


class snd_ns_data_prepocessor(object):
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def __chunkify(self, line, chunk_size):
        result = []
        if chunk_size <= len(line):
            result.extend([line[:chunk_size]])
            # recursive call to iterate over length of line until end of line is reached
            result.extend(self.__chunkify(line[chunk_size:], chunk_size))
        elif line:  # termination statement to append last bit of line.
            result.extend([line])
        return result

    def preprocess_and_save_snd_data(self):
        for snd_file in ['snd-cert', 'snd-unm']:
            snd_file_dir = 'negative-selection/syscalls/{}/'.format(snd_file)  # loop over both cert and unm dirs
            # Create chunks of size N from training file:
            with open(snd_file_dir + '{}.train'.format(snd_file)) as train_file:
                train = []
                for train_line in train_file:
                    train_line = train_line.strip()
                    train += self.__chunkify(train_line, self.chunk_size)
                train_file.close()

            # Save the generated chunkified training set to file
            if not os.path.exists(snd_file_dir + 'N{}'.format(self.chunk_size)):
                os.makedirs(snd_file_dir + 'N{}'.format(self.chunk_size))
            with open(snd_file_dir + 'N{}/{}.train'.format(self.chunk_size, snd_file), "w") as train_save_file:
                for train_line in train:
                    train_save_file.write(train_line + '\n')
                train_save_file.close()

            # Loop all three test and label files respectively
            for i in range(3):
                # Create chunks of size N from test files while mapping label indexes for each chunk to corresponding
                # label:
                with open(snd_file_dir + '{}.{}.test'.format(snd_file, i + 1)) as test_file, \
                        open(snd_file_dir + '{}.{}.labels'.format(snd_file, i + 1)) as labels_file, \
                        open(snd_file_dir + 'N{}/{}.{}.test'.format(self.chunk_size, snd_file, i + 1),
                             'w') as test_save_file, \
                        open(snd_file_dir + 'N{}/{}.{}.labels'.format(self.chunk_size, snd_file, i + 1),
                             'w') as labels_save_file:

                    for i, (test_line, labels_line) in enumerate(zip(test_file, labels_file)):
                        test_line = test_line.strip()
                        test_substrings = self.__chunkify(test_line, self.chunk_size)
                        if len(test_substrings[-1]) == self.chunk_size:
                            for chunk in test_substrings:
                                test_save_file.write(chunk + '\n')
                                labels_save_file.write(str(i) + '\n')  # write reference to label index to separate

                        else:
                            test_substrings[-1] = test_substrings[-1] + "-" * (
                                        self.chunk_size - len(test_substrings[-1]))
                            if any([len(s) != self.chunk_size for s in test_substrings]):
                                raise AttributeError("testsubstring chunks are not of size {}".format(self.chunk_size))
                            for chunk in test_substrings:
                                test_save_file.write(chunk + '\n')
                                labels_save_file.write(str(i) + '\n')

                    test_file.close()
                    labels_file.close()
                    test_save_file.close()
                    labels_save_file.close()
