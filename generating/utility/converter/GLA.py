import numpy as np
import librosa


def window_rms(a, window_size):
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'valid'))


def rms_corr(y1, y2):
    rms1 = window_rms(y1, 3)
    rms2 = window_rms(y2, 3)
    return np.corrcoef(rms1, rms2)[0][1]


def his_mae(y1, y2):
    mae1 = np.mean(np.abs(y1 - y2))
    mae2 = np.mean(np.abs(y1 + y2))
    return np.minimum(mae1, mae2)


def align(y1, y2, max_offset=10):
    if np.mean(np.abs(y1-y2))>np.mean(np.abs(y1+y2)): y2 = -y2
    mae = []
    for i in range(-max_offset, max_offset+1):
        if i<0: mae.append(np.mean(np.abs(y1[-i:] - y2[:i])))
        elif i>0: mae.append(np.mean(np.abs(y1[:-i] - y2[i:])))
        elif i==0: mae.append(his_mae(y1, y2))
    ind = np.argmin(mae) - max_offset
    if ind<0: return np.concatenate((y1[:-ind], y2[:ind]))
    if ind>0: return np.concatenate((y2[ind:], y1[-ind:]))
    return y2


def GLA(S,
        res,
        n_iter       = 1000,
        y_true       = None,
        return_y     = False,
        log_at       = None,
        random_state = None,
        y_last       = None,
        length       = 8000
        ):
    
    if res==128:
        tail = np.zeros((129, 128))
        S = np.concatenate((S, tail), axis=0) + 1e-16
        hop_length, win_length, n_fft = 64, 256, 512
    elif res==256:
        tail = np.zeros((257, 256))
        S = np.concatenate((S, tail), axis=0) + 1e-16
        hop_length, win_length, n_fft = 32, 256, 1024
        
    if log_at is None:
        log_at = get_log_list(n_iter)
        
    if random_state is None:
        rng = np.random
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)
    
    if (y_last is not None) and (res == 128):
        y = np.concatenate((y_last.copy(), np.zeros((128,))), 0)
        S0 = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        angles = S0 / np.abs(S0)
    else:
        angles = np.empty(S.shape, dtype=np.complex64)
        angles[:] = np.exp(2j * np.pi * rng.rand(*S.shape))

    rebuilt = 0.0
    maes, rmss = [], []
    for i in range(n_iter):
        tprev = rebuilt
        inverse = librosa.istft(S * angles, hop_length=hop_length, win_length=win_length)
        rebuilt = librosa.stft(inverse, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        angles[:] = rebuilt - (0.99 / (1 + 0.99)) * tprev
        angles[:] /= np.abs(angles) + 1e-16

        if not return_y:
            if (i+1) in log_at:
                y = librosa.istft(S * angles, hop_length=hop_length, win_length=win_length)
                maes.append( his_mae(y_true, align(y_true, y[:length])) )
                rmss.append( rms_corr(y_true, align(y_true, y[:length])) )
            
    if return_y:
        y = librosa.istft(S * angles, hop_length=hop_length, win_length=win_length)
        y = y[:length] / np.max(np.abs(y[:length]))
        return y[:length]
    else:
        return np.array(maes), np.array(rmss)
    

def get_log_list(n_iter):
    # print(n_iter)
    assert((n_iter > 100)&(n_iter%100==0))
    log_at = np.arange(10, 100, 10).tolist()
    log_at += np.arange(100, np.minimum(n_iter, 1000)+1, 100).tolist()
    if n_iter >= 1000:
        log_at += np.arange(1000, np.minimum(n_iter, 5000)+1, 200).tolist()
        if n_iter >= 5000:
            log_at += np.arange(5000, np.minimum(n_iter, 20000)+1, 500).tolist()
    return np.unique(log_at).tolist()