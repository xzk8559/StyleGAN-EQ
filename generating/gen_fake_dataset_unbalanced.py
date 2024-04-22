import os
import torch
import pickle
import librosa

import scipy.io
import numpy as np
import utility.spectrogram as spc

from PIL import Image
from tqdm import tqdm


'''
Create an unbalanced fake dataset
-------------------------------------------------------------------------------
1. Generate latent vectors and spectrograms with StyleGAN.
2. Reconstruct time histories with GLA.
3. Calculate scales of sprctrograms when PGA = 1.0 m/s^2.
4. Calculate response spectra.
-------------------------------------------------------------------------------
Please generate in batches in practice!
'''


def step1(num, model_path, dset_path, bs=32):
    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    
    z = torch.randn([num, G.z_dim]).cuda()
    w = spc.mapping_batch(num, z, G, b=bs)
    s = spc.synthesis_batch(num, w, G, b=bs)
    
    w = w.cpu().numpy()[:, 0, :]
    s = s.cpu().numpy().squeeze()
    s = spc.norm_post_synthesis(s)
    np.save(dset_path + '/latent_vector/w.npy', w)
        
    for i in range(num):
        image = s[i].copy()
        image = image / np.max(image) * 255
        image = Image.fromarray(image.astype('uint8'))
        
        n = str(i)
        for _ in range(5-len(str(i))): n = '0' + n
        image.save(dset_path + '/spec128/images/' + n + '.png')
        if i%1000 == 0:
            print(i)
    return


def GLA_sim(S, niter=3000):
    rng = np.random
    tail = np.zeros((129, 128))
    S = np.concatenate((S, tail), axis=0) + 1e-16
    angles = np.empty(S.shape, dtype=np.complex64)
    angles[:] = np.exp(2j * np.pi * rng.rand(*S.shape))
    rebuilt = 0.0
    for i in range(niter):
        tprev = rebuilt
        inverse = librosa.istft(S * angles, hop_length=64, win_length=256)
        rebuilt = librosa.stft(inverse, n_fft=512, hop_length=64, win_length=256)
        angles[:] = rebuilt - (0.99 / (1 + 0.99)) * tprev
        angles[:] /= np.abs(angles) + 1e-16
    y = librosa.istft(S * angles, hop_length=64, win_length=256)[:8000]
    return y / np.max(np.abs(y))

    
def step2(num, start, dset_path):
    hpath = dset_path + '/history/'
    spath = dset_path + '/spec128/images/'
    h = []
    flist = os.listdir(spath)
    flist.sort()
    for i in tqdm(range(start, start + num)):
        s = Image.open(spath + flist[i])
        s = np.array(s) / 255.
        h.append(GLA_sim(s, 3000))
    h = np.array(h)
    np.save(hpath + 'hist-{}-{}.npy'.format(start, start + num), h)
    scipy.io.savemat(hpath + 'hist-{}-{}.mat'.format(start, start + num), {'history': h})
    return


def STFT_sim(hist):
    hist = np.concatenate((hist, np.zeros((128,))))+ 1e-16
    spec = librosa.stft(hist, 512, win_length=256)
    return np.abs(spec)[:128]


def step3(dset_path):
    scale = []
    h = np.load(dset_path + '/history/hist.npy')
    for i in tqdm(range(h.shape[0])):
        s_h = STFT_sim(h[i], 128)
        scale.append(np.max(s_h))
    np.save(dset_path + '/spec128/scale.npy', scale)
    return
    

def save_rspec_npy(dset_path):
    fpath = dset_path + '/rspectra/0.05-2.00/'
    h = scipy.io.loadmat(fpath + 'rspec.mat')['rspec']
    np.save(fpath + 'rspec.npy', h)
    return


def step4(dset_path):
    print('Run "cal_rspectra.m" in Matlab!')
    save_rspec_npy(dset_path)
    return


# Example
# num = 10000
# dset_path = '../../data/kiknet-fake/unbalanced'
# model_path = '../pretrained_models/stylegan3-kiknet-15k.pkl'

# step1(num, model_path, dset_path, bs=32)
# step2(num, 0, dset_path)
# step3(dset_path)
# step4(dset_path)
