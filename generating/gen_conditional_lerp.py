import scipy
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import utility.spectrogram as spc

from tqdm import tqdm
from scipy.fft import fft
from utility.converter.GLA import GLA


def gen_cond_lerp(
        seed1         = 0,
        seed2         = 1,
        frames        = 1,
        Gpath         = None,
        spath         = None,
        preview_seeds = False
        ):
    with open(Gpath, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    
    print('Interpolating seed {} and seed {}...'.format(seed1, seed2))
    z1 = np.random.RandomState(seed1).randn(G.z_dim)
    z2 = np.random.RandomState(seed2).randn(G.z_dim)
    z = torch.from_numpy(np.stack([z1, z2])).cuda()
    w = G.mapping(z, None)
    
    x = np.arange(-4, 6)
    y = np.tile(w.cpu().numpy(), [5, 1, 1])
    interp = scipy.interpolate.interp1d(x, y, kind='cubic', axis=0)
    
    s_lerp, w_lerp = [], []
    for frame_idx in tqdm(range(2 * frames)):
        wi = torch.from_numpy(interp(frame_idx / frames)).cuda().unsqueeze(0)
        si = G.synthesis(wi, noise_mode='const')[0]
        s_lerp.append(si.cpu().numpy())
        w_lerp.append(wi.cpu().numpy())
    
    s_lerp = np.stack(s_lerp).squeeze()
    w_lerp = np.stack(w_lerp).squeeze()
    s_lerp = spc.norm_post_synthesis(s_lerp)
    
    if preview_seeds:
        preview(seed1, seed2, s_lerp)
        return
    else:
        np.save(spath + '/w_{}_{}.npy'.format(seed1, seed2), w_lerp)
        np.save(spath + '/s_{}_{}.npy'.format(seed1, seed2), s_lerp)
        return s_lerp, w_lerp


def preview(seed1, seed2, s):
    plt.figure(tight_layout=True, figsize=(8, 4), dpi=300)
    plt.subplot(1, 2, 1)
    plt.imshow(s[0])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Seed-{}'.format(seed1), fontsize=24)
    plt.subplot(1, 2, 2)
    plt.imshow(s[s.shape[0]//2])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Seed-{}'.format(seed2), fontsize=24)
    plt.show()
    return


def compute_history_lerp(s, seed1, seed2, max_iter, spath):
    h = []
    print('Processing {} frames...'.format(s.shape[0]))
    for i in tqdm(range(s.shape[0])):
        if i==0:
            h.append(GLA(s[i], 128, 5000, return_y=True))
        else:
            h1 = h[-1].copy()
            h2 = GLA(s[i], 128, max_iter, y_last=h1, return_y=True)
            h2 = -h2 if np.mean(np.abs(h1-h2))>np.mean(np.abs(h1+h2)) else h2
            h.append(h2)
    h = np.stack(h)
    np.save(spath + '/h_{}_{}.npy'.format(seed1, seed2), h)
    return h


def render_video(s, h, seed1, seed2, spath, interval=60):
    import matplotlib.gridspec as gridspec
    import matplotlib.animation as animation
    ft = np.abs(fft(h))[:, :2000]
    
    fig = plt.figure(figsize=(24,6), tight_layout=True)
    gs = gridspec.GridSpec(1, 4)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xticks([])
    ax1.set_yticks([])
    img = ax1.imshow(s[0])
    
    ax2 = fig.add_subplot(gs[0, 1:3])
    ymax = np.max(np.abs(h))
    ax2.set_ylim(-ymax, ymax)
    ax2.set_xticks([])
    ax2.set_yticks([])
    line1, = ax2.plot(h[0])
    
    ax3 = fig.add_subplot(gs[0, 3])
    ymax = np.max(ft)
    ax3.set_ylim(0, ymax)
    ax3.set_xticks([])
    ax3.set_yticks([])
    line2, = ax3.plot(ft[0])
    
    def animate(i):
        img.set_array(s[i])
        line1.set_ydata(h[i])
        line2.set_ydata(ft[i])
        return img, line1, line2,
    ani = animation.FuncAnimation(
        fig, animate, interval=interval, blit=True, save_count=s.shape[0])
    ani.save(spath + '/video_{}_{}.gif'.format(seed1, seed2))
    return


seed1 = 820218
seed2 = 820778
frames = 120
GLA_iter = 2000
save_path = 'save/lerp'
model_path = '../pretrained_models/stylegan3-kiknet-15k.pkl'

# gen_cond_lerp(seed1, seed2, model_path, preview_seeds=True)
spectrogram, _ = gen_cond_lerp(seed1, seed2, frames, model_path, save_path)

history = compute_history_lerp(spectrogram, seed1, seed2, GLA_iter, save_path)
render_video(spectrogram, history, seed1, seed2, save_path, interval=40)

