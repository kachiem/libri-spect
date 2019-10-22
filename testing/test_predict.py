import librispect as lspct
from librispect.features import spectrogram, predict
import glob
import librosa
import numpy as np
import pytest
from librispect.utils import split_validation

SPECT_HEIGHT = 2048
STFT_HEIGHT = 2048
WINDOW_SIZE = 16
BATCH_SIZE = 32
STEP_SIZE = 4
TERMS = 8
PREDICT_TERMS = 8


def test_spect_predict_maker():
    '''integration testing of prediction generation'''
    hparams = lspct.features.spectrogram.HPARAMS
    test_spect_predict_maker = predict.spect_predict_maker(hparams, terms=TERMS, predict_terms=PREDICT_TERMS)
    path_list = glob.glob((lspct.paths.WAV_DIR / '*.wav').as_posix())[0:1] 
    print("path_list: ", path_list)
    pred_spect = test_spect_predict_maker.batch_iter(path_list, 2)
    #neg_batch, pos_batch, labels, idxs = test_spect_predict_maker.batch_iter(path_list, 2)
    if len(path_list) <= 1:
        # term_batch, pterm_batch, labels, idxs
        print("test_spect_predict_maker: ", test_spect_predict_maker)
        print("pred_spect: ", pred_spect)
        batch_size = test_spect_predict_maker.batch_ss_per_epoch(
            path_list, 32)  # turn batch_set into param for epoch
        # if (actual batch_per_epoch(batch_size) == expected batch_size)
        assert batch_size % 2 == 0


def test_split_validation():
    path_list = list(lspct.paths.WAV_DIR.glob("*.wav"))[0:10]
    train_path_list, val_path_list = split_validation(path_list, 0.1)
        batch_size = test_spect_predict_maker.batch_ss_per_epoch(path_list, 32)  # turn batch_set into param for epoch
        # if (actual batch_per_epoch(batch_size) == expected batch_size)
        if batch_size % 2 == 0:  # expected batch size:
            train_path_list, val_path_list = split_validation(test_spect_predict_maker, 0.1)
            return print("training: ", train_path_list, "\n", "validation: ", val_path_list)
        else:
            return "Uneven batch size"


if __name__ == "__main__":
    test_spect_predict_maker()
