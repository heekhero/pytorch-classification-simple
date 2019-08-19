from torch.utils.data.sampler import Sampler

import numpy as np

class DataSampler(Sampler):

    def __init__(self, inds, bs):
        self._inds = inds
        self._num_data = len(self._inds)
        self._bs = bs
        self._num_batch = int(np.ceil(self._num_data / self._bs))

    def __iter__(self):
        bs = self._bs
        inds_batch = np.empty(shape=(self._num_batch - 1, bs)).astype(np.int)
        rand_inds = np.random.permutation(self._num_batch - 1)
        for i in range(self._num_batch - 1):
            ind_batch = self._inds[i*bs: (i+1)*bs]
            inds_batch[rand_inds[i], :] = ind_batch
        inds_batch = inds_batch.reshape(-1)
        inds_batch = np.append(inds_batch, self._inds[(self._num_batch-1) * bs: self._num_data])
        return iter(inds_batch)

    def __len__(self):
        return self._num_data