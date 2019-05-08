import librispect as lspct
from librispect.features import spectrogram, predict
import glob
<<<<<<< HEAD
import librosa
import numpy as np
import pytest
from utils import split_validation
=======
>>>>>>> 3123cbe61362c898ba13f39aa35d4c94c8067606

SPECT_HEIGHT = 2048
STFT_HEIGHT = 2048
WINDOW_SIZE = 16
BATCH_SIZE = 32
STEP_SIZE = 4
TERMS = 8
PREDICT_TERMS = 8


def test_spect_predict_maker():
    '''integration testing of prediction generation'''
<<<<<<< HEAD
    hparams = lspct.features.spectrogram.HPARAMS
    test_spect_predict_maker = predict.spect_predict_maker(hparams, terms=TERMS, predict_terms=PREDICT_TERMS)
    path_list = glob.glob((lspct.paths.WAV_DIR / '*.wav').as_posix())[0:1] 
=======
    hparams = spectrogram.HPARAMS
    test_spect_predict_maker = predict.spect_predict_maker(
        hparams, terms=TERMS, predict_terms=PREDICT_TERMS)
    path_list = glob.glob((lspct.paths.WAV_DIR / '*.wav').as_posix())[:2]
>>>>>>> 3123cbe61362c898ba13f39aa35d4c94c8067606
    pred_spect = test_spect_predict_maker.batch_iter(path_list, 2)

    # term_batch, pterm_batch, labels, idxs
    batch_size = batch_ss_per_epoch(pred_spect, 32);  # turn batch_set into param for epoch
    # if (actual batch_per_epoch(batch_size) == expected batch_size)
    if batch_size % 2 == 0:  # expected batch size:
        # run batch_per_epoch? then create training/val sets
<<<<<<< HEAD
        train_path_list, val_path_list = split_validation(path_list, 0.1)
        return print("training: ", train_path_list,"\n" ,"validation: ", val_path_list)
=======
        train_path_list, val_path_list = test_spect_predict_maker.split_validation(path_list, 0.1)
        return print("training: ", train_path_list, "\n", "validation: ", val_path_list)
>>>>>>> 3123cbe61362c898ba13f39aa35d4c94c8067606
    else:
        return "Uneven batch size"


if __name__ == "__main__":
    test_spect_predict_maker()
