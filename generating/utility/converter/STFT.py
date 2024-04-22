import numpy as np
import librosa


def STFT(his, res):
    assert len(his)==8000
    if res==128:
        his = np.concatenate((his, np.zeros((128,))))+ 1e-16
        spec = librosa.stft(his, 512, win_length=256)
        spec = np.abs(spec)[:128]
        # freqs = librosa.fft_frequencies(sr=50, n_fft=512)
    elif res==256:
        his = np.concatenate((his, np.zeros((160,))))+ 1e-16
        spec = librosa.stft(his, 1024, win_length=256, hop_length=32)
        spec = np.abs(spec)[:256]
        # freqs = librosa.fft_frequencies(sr=50, n_fft=1024)
    return spec
