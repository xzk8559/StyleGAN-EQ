import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import utility.spectrogram as spc

from tqdm import tqdm
from scipy.fft import fft
from utility.converter.GLA import GLA


def gen_uncond(num, Gpath):
    with open(Gpath, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    
    z = torch.randn([num, G.z_dim]).cuda()
    w = G.mapping(z, None)
    s = G.synthesis(w, noise_mode='const')
    
    w = w.cpu().numpy()
    s = s.cpu().numpy().squeeze()
    s = spc.norm_post_synthesis(s)
    return s, w


def plots(s, h, ffts, plot_n):
    nfigs = s.shape[0]//plot_n + 1
    if s.shape[0] % plot_n > 0:
        nfigs += 1
        
    for n in range(nfigs):
        sta = plot_n * n
        end = np.minimum(plot_n*(n+1), s.shape[0])
        sn = s[sta:end]
        h_recn = h[sta:end]
        fft_recn = ffts[sta:end]
        
        plt.figure(tight_layout=True, figsize=(16, 4*plot_n))
        for i in range(sn.shape[0]):
            plt.subplot(plot_n, 4, i*4+1)
            plt.imshow(sn[i])
            plt.xticks([])
            plt.yticks([])
            plt.ylabel('EQ-{}'.format(sta+i), fontsize=36)
            
            plt.subplot(plot_n, 4, (i*4+2,i*4+3))
            plt.plot(h_recn[i])
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(plot_n, 4, i*4+4)
            plt.plot(fft_recn[i])
            plt.xticks([])
            plt.yticks([])
        plt.show
    return


def show_history(s, max_iter, plot=False, plot_n=4):
    hist, ffts = [], []
    for i in tqdm(range(s.shape[0])):
        x = GLA(s[i], 128, max_iter, return_y=True)
        hist.append(x)
        ffts.append(np.abs(fft(x))[:2000])
    if plot: plots(s, hist, ffts, plot_n)
    return np.array(hist), np.array(ffts)


gen_num = 4
GLA_iter = 1000
model_path = '../pretrained_models/stylegan3-kiknet-15k.pkl'

spectrogram, _ = gen_uncond(num=gen_num, Gpath=model_path)
history,     _ = show_history(spectrogram, GLA_iter, plot=True, plot_n=4)

