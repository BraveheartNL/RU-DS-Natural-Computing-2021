
from pandas.core.frame import DataFrame
import numpy as np


class batch_generator:

    @staticmethod
    def get_batch(data, labels, batch_size: int) -> DataFrame:
        """Generates unlimited randomly shuffled batches of data and label pairs."""
        if len(data) != len(labels):
            raise AttributeError("Labels and data frames are of unequal size!")

        indices = np.arrange(len(data))
        batch = []
        while True:
            np.random.shuffle(indices)
            for i in indices:
                batch.append(i)
                if len(batch) == batch_size:
                    yield data[batch], labels[batch]
                    batch = []
