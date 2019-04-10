from deepspectrograminversion.features import spectrogram
import glob
import deepspectrograminversion as dsi
import librosa
import numpy as np
import pytest


def test_stft_to_wav():
    """testing the similarity between the original and reconstructed data"""
    hparams = dsi.features.spectrogram.HPARAMS
    wavfile = glob.glob((dsi.paths.WAV_DIR / '*.wav').as_posix())[0]
    data, hparams['sample_rate'] = librosa.load(wavfile)
    stft = spectrogram.wav_to_stft(data, hparams)
    reconstructed_data = spectrogram.stft_to_wav(stft, hparams)

    # assume aligned, trim end
    min_len = min(len(data), len(reconstructed_data))
    assert np.allclose(data[:min_len], reconstructed_data[:min_len])


def test_mel_spectrogram():
    """test the mel sepctrogram"""
    hparams = dsi.features.spectrogram.HPARAMS
    wavfile = glob.glob((dsi.paths.WAV_DIR / '*.wav').as_posix())[0]
    data, hparams['sample_rate'] = librosa.load(wavfile)
    mel_basis = spectrogram.build_mel_basis(hparams)
    for pre in [True, False]:
        stft = spectrogram.wav_to_stft(data, hparams, pre=pre)
        spect = spectrogram.spectrogram(stft, hparams)
        assert len(spect.shape) == 2, spect.shape
        mel_spect = spectrogram.linear_to_mel(spect, mel_basis)
        assert len(mel_spect.shape) == 2, mel_spect.shape
        assert hparams['num_mels'] in mel_spect.shape, mel_spect.shape
        assert len(mel_spect) != 0


def test_spect_iter():
    """test the spect_maker class with the diff
    between spect created by class and function"""
    hparams = dsi.features.spectrogram.HPARAMS
    wavfile = glob.glob((dsi.paths.WAV_DIR / '*.wav').as_posix())[0]
    data, hparams['sample_rate'] = librosa.load(wavfile)
    sm = spectrogram.spect_maker(hparams)
    for spect, stft in sm.spect_iter(glob.glob((dsi.paths.WAV_DIR /
                                                '*.wav').as_posix())[:25]):
        assert len(stft.shape) == 3, stft.shape
        assert stft.shape[0] == 2, stft.shape
        assert len(spect.shape) == 2, spect.shape
        new_spect = spectrogram.spectrogram(stft, hparams)
        assert len(new_spect.shape) == 2, new_spect.shape
        assert np.allclose(spect, new_spect)


def test_mel_spect_iter():
    hparams = dsi.features.spectrogram.HPARAMS
    wavfile = glob.glob((dsi.paths.WAV_DIR / '*.wav').as_posix())[0]
    data, hparams['sample_rate'] = librosa.load(wavfile)
    sm = spectrogram.spect_maker(hparams)
    mel_basis = spectrogram.build_mel_basis(hparams)
    for mspect, stft in sm.mel_spect_iter(glob.glob((dsi.paths.WAV_DIR /
                                                     '*.wav').as_posix())[:25]):
        assert len(stft.shape) == 3, stft.shape
        assert stft.shape[0] == 2, stft.shape
        assert len(mspect.shape) == 2, mspect.shape
        new_spect = spectrogram.spectrogram(stft, hparams)
        assert len(new_spect.shape) == 2, new_spect.shape
        new_mspect = spectrogram.linear_to_mel(new_spect, mel_basis)
        assert np.allclose(mspect, new_mspect)


test_params = [
    (1, 16, 1, 32),
    (5, 16, 1, 32),
    (5, 4, 1, 32),
    (5, 32, 1, 32),
    (5, 16, 4, 32),
    (5, 16, 16, 32),
    (5, 16, 1, 8),
    (5, 16, 1, 64),
    (25, 16, 1, 32),
    (100, 16, 1, 32),
    # (None, 16, 16, 32),
]


@pytest.mark.parametrize("samples,window_size,step_size,batch_size", test_params)
def test_batch_ss_iter(samples, window_size, step_size, batch_size):
    hparams = dsi.features.spectrogram.HPARAMS
    wavfile = glob.glob((dsi.paths.WAV_DIR / '*.wav').as_posix())[0]
    data, hparams['sample_rate'] = librosa.load(wavfile)
    sm = spectrogram.spect_maker(
        hparams, window_size=window_size, step_size=step_size)
    path_list = glob.glob((dsi.paths.WAV_DIR / '*.wav').as_posix())
    if samples:
        path_list = path_list[:samples]
    steps_per_epoch = sm.batch_ss_per_epoch(path_list, batch_size)
    assert steps_per_epoch > 0, steps_per_epoch
    for i, (spect_batch, stft_batch) in enumerate(sm.batch_iter(path_list, batch_size, False)):
        assert spect_batch.shape[0] == stft_batch.shape[
            0], (spect_batch.shape[0], stft_batch.shape[0])
        for batch in [spect_batch, stft_batch]:
            assert batch.shape[0] <= batch_size, (batch.shape[0], batch_size)
            assert batch.shape[
                -1] == window_size, (batch.shape[-1], window_size)
            if step_size < window_size:
                overlap = window_size - step_size
                first_sample = batch[0, ...]
                second_sample = batch[1, ...]
                assert np.allclose(
                    first_sample[..., -overlap:], second_sample[..., :overlap])
        if i == 0:
            start_spect_batch = spect_batch
            start_stft_batch = stft_batch
        elif i == steps_per_epoch:
            assert np.allclose(start_spect_batch,
                               spect_batch), "hasn't looped around yet"
            assert np.allclose(
                start_stft_batch, stft_batch), "hasn't looped around yet"
            break


if __name__ == "__main__":
    test_stft_to_wav()
    test_mel_spectrogram()
    test_spect_iter()
    test_mel_spect_iter()
