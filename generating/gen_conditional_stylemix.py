import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import utility.spectrogram as spc


def gen_cond_stylemix_pair(n, seed1, seed2, layers, G_path, plot_mode='spec', dpi=300):
    '''
    Style mixing n-samples with a target sample on m groups of layers
    -----------------------------------------------------------------
    Input:
        seed1: seed to generate n source samples
        seed2: seed to generate 1 target sample
        layers = [group1, group2, group3 ... groupm]
            e.g. groupm = [0, 2, 5 ...]
    Output:
        spectrogram of source samples, (   n, 128, 128)
        spectrogram of target sample,  (   1, 128, 128)
        spectrogram mixed,             (m, n, 128, 128)
    -----------------------------------------------------------------
    '''
    
    assert len(layers)>0
    assert len(layers[0])>0
    assert plot_mode in ['spec', 'time', 'freq']
    
    with open(G_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    
    z_org = np.random.RandomState(seed1).randn(n, G.z_dim)
    z_trg = np.random.RandomState(seed2).randn(1, G.z_dim)
    z_org = torch.from_numpy(z_org).cuda()
    z_trg = torch.from_numpy(z_trg).cuda()
    w_org = G.mapping(z_org, None, truncation_psi=1.0)
    w_trg = G.mapping(z_trg, None, truncation_psi=1.0)
    w_org[:, 0, :] = w_trg[:, 0, :]
    
    s_mixes = []
    for groupm in layers:
        w_mix = w_org.clone().detach()
        for ind in groupm:
            w_mix[:, ind, :] = w_trg[:, ind, :]
        s_mix = G.synthesis(w_mix, noise_mode='const').cpu().numpy()
        s_mixes.append(s_mix)
    
    s_org = G.synthesis(w_org, noise_mode='const').cpu().numpy()
    s_trg = G.synthesis(w_trg, noise_mode='const').cpu().numpy()
    s_mix = np.stack(s_mixes, 0)
    
    s_org = spc.norm_post_synthesis(s_org).squeeze(1)
    s_trg = spc.norm_post_synthesis(s_trg).squeeze(1)
    s_mix = spc.norm_post_synthesis(s_mix).squeeze(2)
    
    print('s_org:', s_org.shape)
    print('s_trg:', s_trg.shape)
    print('s_mix:', s_mix.shape)
    show_mixed_pair(s_org, s_trg, s_mix, plot_mode, layers, dpi=dpi)
    return s_org, s_trg, s_mix
    

def show_mixed_pair(s_org, s_trg, s_mix, plot_mode, layers, dpi):
    # plot first 8 results for test
    figw = (s_mix.shape[0] + 2)
    figh = np.minimum(8, s_mix.shape[1])
    
    plt.figure(tight_layout=True, figsize=(figw*4, figh*4), dpi=dpi)
    for yi in range(figh):
        n_start = yi * figw
        
        plt.subplot(figh, figw, n_start+1)
        plt.imshow(s_org[yi])
        plt.xticks([])
        plt.yticks([])
        if yi == (figh - 1): plt.xlabel('Source', fontsize=24)
        
        for xi in range(0, figw-2):
            plt.subplot(figh, figw, n_start+xi+2)
            if plot_mode=='spec':
                plt.imshow(s_mix[xi, yi])
            elif plot_mode=='freq':
                plt.plot(np.mean(s_org[yi], 1),     color='C0')
                plt.plot(np.mean(s_mix[xi, yi], 1), color='C2')
                plt.plot(np.mean(s_trg[0], 1),      color='C1')
            elif plot_mode=='time':
                plt.plot(np.mean(s_org[yi], 0),     color='C0')
                plt.plot(np.mean(s_mix[xi, yi], 0), color='C2')
                plt.plot(np.mean(s_trg[0], 0),      color='C1')
            plt.xticks([])
            plt.yticks([])
            xlabel = [str(nl) for nl in layers[xi]]
            if yi == (figh - 1):
                plt.xlabel(','.join(xlabel), fontsize=24)
            
        plt.subplot(figh, figw, n_start+figw)
        plt.imshow(s_trg[0])
        plt.xticks([])
        plt.yticks([])
        if yi == (figh - 1): plt.xlabel('Target', fontsize=24)
    return


def gen_cond_stylemix_square(n, m, seed1, seed2, G_path, dpi=300):
    '''
    Style mixing n-samples with another n-samples on layers [1-8/9-15]
    ------------------------------------------------------------------
    Input:
        seed1: seed to generate n source samples (layer 0-8)
        seed2: seed to generate m target samples (layer 9-15)
    Output:
        spectrogram of source samples, (   n, 128, 128)
        spectrogram of target sample,  (   1, 128, 128)
        spectrogram mixed,             (m, n, 128, 128)
    ------------------------------------------------------------------
    '''
    
    with open(G_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    
    z_org = np.random.RandomState(seed1).randn(n, G.z_dim)
    z_trg = np.random.RandomState(seed2).randn(m, G.z_dim)
    z_org = torch.from_numpy(z_org).cuda()
    z_trg = torch.from_numpy(z_trg).cuda()
    w_org = G.mapping(z_org, None, truncation_psi=1.0)
    w_trg = G.mapping(z_trg, None, truncation_psi=1.0)
    
    s_mixes = []
    for mi in range(m):
        w_mix = w_org.clone().detach()
        w_mix[:, 9:, :] = w_trg[mi:mi+1, 9:, :]
        s_mix = G.synthesis(w_mix, noise_mode='const').cpu().numpy()
        s_mixes.append(s_mix)
    
    s_org = G.synthesis(w_org, noise_mode='const').cpu().numpy()
    s_trg = G.synthesis(w_trg, noise_mode='const').cpu().numpy()
    s_mix = np.stack(s_mixes, 0)
    
    s_org = spc.norm_post_synthesis(s_org).squeeze(1)
    s_trg = spc.norm_post_synthesis(s_trg).squeeze(1)
    s_mix = spc.norm_post_synthesis(s_mix).squeeze(2)
    
    print('s_org:', s_org.shape)
    print('s_trg:', s_trg.shape)
    print('s_mix:', s_mix.shape)
    show_mixed_square(s_org, s_trg, s_mix, dpi=dpi)
    return s_org, s_trg, s_mix


def show_mixed_square(s_org, s_trg, s_mix, dpi):
    figw = s_mix.shape[0] + 1
    figh = s_mix.shape[1] + 1
    
    plt.figure(tight_layout=True, figsize=(figw*4, figh*4), dpi=dpi)
    # source
    for yi in range(1, figh):
        n_start = yi * figw + 1
        plt.subplot(figh, figw, n_start)
        plt.imshow(s_org[yi-1])
        plt.xticks([])
        plt.yticks([])
    # target
    for xi in range(1, figw):
        plt.subplot(figh, figw, xi+1)
        plt.imshow(s_trg[xi-1])
        plt.xticks([])
        plt.yticks([])
    # mixed
    for yi in range(1, figh):
        n_start = yi * figw + 1
        for xi in range(1, figw):
            plt.subplot(figh, figw, n_start+xi)
            plt.imshow(s_mix[xi-1, yi-1])
            plt.xticks([])
            plt.yticks([])
    return


seed1 = 820218
seed2 = 820778
model_path = '../pretrained_models/stylegan3-kiknet-15k.pkl'
layers = [ [0,9,10,11,12,13,14,15], [0,1,2,3,4,5,6,7,8] ]

_ = gen_cond_stylemix_pair(3, seed1, seed2, layers, model_path, 'spec', dpi=75)
_ = gen_cond_stylemix_square(3, 3, seed1, seed2, model_path, dpi=75)

