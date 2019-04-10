import numpy as np
import librosa
from scipy.signal import lfilter

HPARAMS = {
    # filtering
    "highcut": 3000,
    "lowcut": 10,
    # spectrograms
    "mel_filter": True,  # should a mel filter be used?
    "num_mels": 16,  # how many channels to use in the mel-spectrogram
    "num_freq": 2048,  # how many channels to use in a spectrogram
    "sample_rate": 22050,  # what rate are your WAVs sampled at?
    "preemphasis": 0.97,
    "frame_shift_ms": 20,  # step size for fft
    "frame_length_ms": 50,  # frame length for fft
    "min_level_db": -100,  # minimum threshold db for computing spe
    "ref_level_db": 20,  # reference db for computing spec
    "fmin": 10,  # low frequency cutoff for mel filter
    "fmax": 15000,  # high frequency cutoff for mel filter
    # spectrogram inversion
    "max_iters": 200,
    "griffin_lim_iters": 20,
    "power": 1.5,
}


def get_params(hparams):
    """A method that computes the parameters for stft transform

    Return the FFT window size, number audio of frames bwteen STFT columns
    and length for each window

    Window length is defined by:
    Each frame of the audio is passed through a window function
    with a certain length

    Args:
        hparams: a list of parameters for the stft including:
        frequency, step size for fft, frame length for fft,
        sample rate and filtering levels

    Returns:
        n_fft: FFT window size
        hop_length: frames bewtewen windows
        win_length: length for each window with certain window fucntion
    """
    n_fft = (hparams["num_freq"] - 1) * 2
    hop_length = int(hparams["frame_shift_ms"] / 1000 * hparams["sample_rate"])
    win_length = int(hparams["frame_length_ms"] / 1000 * hparams["sample_rate"])
    return n_fft, hop_length, win_length


def stft_to_wav(stft, hparams):
    """inverse the stft to produce the wav file data"""
    _, hop_length, win_length = get_params(hparams)
    return librosa.istft(
        _i_phase_and_magnitude(stft), hop_length=hop_length, win_length=win_length
    )


def wav_to_stft(data, hparams, pre=False):
    """takes in a wav file to produce a stft matrix"""
    n_fft, hop_length, win_length = get_params(hparams)
    if pre:
        data = preemphasis(data, hparams)
    stft_imag = librosa.stft(
        data, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )
    return _phase_and_magnitude(stft_imag)


def _phase_and_magnitude(stft):
    """
    convert from complex stft matrix to phase-magnitude stft matrix
    stft[0,:,:] = magnitude
    stft[1,:,:] = phase
    """
    return np.array((np.abs(stft), np.angle(stft)))


def _i_phase_and_magnitude(stft):
    """convert from phase magnitude stft matrix to complex stft matrix"""
    return stft[0, :, :] * np.exp(1j * stft[1, :, :])


def normalize(stft, hparams):
    """we clip the stft matrix based on filter levels to normalize"""
    return np.clip((stft - hparams["min_level_db"]) / -hparams["min_level_db"], 0, 1)


def _denormalize(S, hparams):
    """restoring stft matrix based on filter levels to denormalize"""
    return (np.clip(S, 0, 1) * -hparams["min_level_db"]) + hparams["min_level_db"]


def _amp_to_db(x):
    """preprocess the matrix for spectrogram computation"""
    return 20 * np.log10(np.maximum(1e-5, x))


def preemphasis(x, hparams):
    """apply low pass butterworth filter to signal"""
    return lfilter([1, -hparams["preemphasis"]], [1], x)


def spectrogram(stft, hparams):
    """spectrogram computation with amp tp db and normalization"""
    spect = _amp_to_db(stft[0, :, :]) - hparams["ref_level_db"]
    return normalize(spect, hparams)


def linear_to_mel(spectrogram, _mel_basis):
    """convert spectrogram from linear-sp to log-spaced mel scale"""
    return np.dot(_mel_basis, spectrogram)


def build_mel_basis(hparams):
    """put mel scale values into nonsquare basis matrix"""
    n_fft = (hparams["num_freq"] - 1) * 2
    return librosa.filters.mel(
        hparams["sample_rate"],
        n_fft,
        n_mels=hparams["num_mels"],
        fmin=hparams["fmin"],
        fmax=hparams["fmax"],
    )


def slicing_window(data, size, step=1):
    """
    A method that slice the given data to numpy array of same shape tensors
    with a certain step size
    Return the sliced tensors in form of a numpy array
    Args:
        data: the data we want to slice
        size: the window size we desire. Should be an integer
        step: the step size of iteration. Should be an integer

    Returns:
        np.array of all the sliced tensors
    """
    sliced = []
    num_elements = data.shape[-1]
    start = range(0, num_elements, step)
    end = range(size, num_elements + 1, step)
    for i, j in zip(start, end):
        sliced.append(data[..., i:j])
    return np.array(sliced)


class spect_maker:
    """
    The Class spect_maker contains iterator that iterate thorugh
    a list of wav files and a generator that return batches of
    feature, label tuple to feed into the model. It also contains
    several helper functions

    Functions:
        __init__: a initializer of class, pass in hparams
        wav_iter: return a generator, takes in path_list and returns
                  the converted numpy array from wav
        batch_ss_per_epoch: returns number of batches for one
                            epoch
        stft_iter: return a generator, takes in path_list and returns
                   the converted stft numpy array
        spect_iter: similar to stft_iter but return the spect numpy
                   array
        mel_spect_iter: return the mel spectrogram instead of the
                        regular spectrogram
        batch_ss_iter: return the generator that produce the feature
                        label pair to feed into the model
    """

    def __init__(self, hparams, window_size=16, step_size=1):
        self.hparams = hparams
        self.window_size = window_size
        self.step_size = step_size
        self.mel_basis = build_mel_basis(hparams)

    def batch_ss_per_epoch(self, path_list, batch_size):
        # calculate number of batches per epoch
        num_batches = 0
        for stft in self.stft_iter(path_list):
            num_batches += np.ceil(
                ((stft.shape[-1] - self.window_size + 1) / self.step_size) / batch_size
            )
        return num_batches

    def wav_iter(self, path_list):
        for wavfile in path_list:
            data, rate = librosa.load(wavfile)
            yield data

    def stft_iter(self, path_list):
        for data in self.wav_iter(path_list):
            yield wav_to_stft(data, self.hparams)

    def spect_iter(self, path_list):
        for stft in self.stft_iter(path_list):
            yield spectrogram(stft, self.hparams), stft

    def mel_spect_iter(self, path_list):
        for spect, stft in self.spect_iter(path_list):
            yield linear_to_mel(spect, self.mel_basis), stft

    def batch_iter(self, path_list, batch_size, shuffle=True):
        while True:
            for spect, stft in self.spect_iter(path_list):
                # slice spect and stft
                spect_sliced = slicing_window(spect, self.window_size, self.step_size)
                stft_sliced = slicing_window(stft, self.window_size, self.step_size)
                if shuffle:
                    ind = np.arange(spect_sliced.shape[0])
                    np.random.shuffle(ind)
                    spect_sliced = spect_sliced[ind]
                    stft_sliced = stft_sliced[ind]
                num_batches = np.ceil(len(spect_sliced) / batch_size)
                for spect_batch, stft_batch in zip(
                    np.array_split(spect_sliced, num_batches),
                    np.array_split(stft_sliced, num_batches),
                ):
                    yield spect_batch, stft_batch
