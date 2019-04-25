import numpy as np
from librispect.features.spectrogram import slicing_window, spect_maker


class spect_predict_maker:
    def __init__(self, hparams, terms=4, predict_terms=4, step_size=1):
        self.hparams = hparams
        self.terms = terms
        self.predict_terms = predict_terms
        self.step_size = step_size
        self.window_size = terms + predict_terms
        self.spect_maker = spect_maker(
            hparams, window_size=self.window_size, step_size=step_size
        )

    def batch_per_epoch(self, path_list, batch_size):
        assert batch_size % 2 == 0, "half positive, half negative examples"
        return self.spect_maker.batch_ss_per_epoch(path_list, batch_size / 2)

    def batch_iter(self, path_list, batch_size):
        while True:
            for spect, _ in self.spect_maker.spect_iter(path_list):
                spect_sliced = slicing_window(spect, self.window_size, self.step_size)
                x_terms = spect_sliced[..., : self.terms]
                x_pterms = spect_sliced[..., -self.predict_terms :]
                neg_terms = np.random.permutation(x_terms)
                neg_pterms = np.random.permutation(x_pterms)

                num_batches = np.ceil(len(spect_sliced) / (batch_size / 2))
                for x_term_batch, x_pterm_batch, neg_term_batch, neg_pterm_batch in zip(
                    *[
                        np.array_split(sliced, num_batches)
                        for sliced in [x_terms, x_pterms, neg_terms, neg_pterms]
                    ]
                ):
                    # keras likes time to be dimension 1
                    term_batch = np.transpose(
                        np.concatenate((x_term_batch, neg_term_batch)), (0, 2, 1)
                    )
                    pterm_batch = np.transpose(
                        np.concatenate((x_pterm_batch, neg_pterm_batch)), (0, 2, 1)
                    )
                    labels = np.concatenate(
                        (
                            np.ones(x_term_batch.shape[0]),
                            np.zeros(neg_term_batch.shape[0]),
                        )
                    )
                    idxs = np.random.permutation(labels.shape[0])
                    yield [term_batch[idxs, ...], pterm_batch[idxs, ...]], labels[
                        idxs, ...
                    ]

    def split_validation(self, path_list, validation_percentage):
        val_index = int(len(path_list) * (1 - validation_percentage))
        val = path_list[val_index:]
        training = path_list[0:(val_index - 1)]
            
        return training, val
    
