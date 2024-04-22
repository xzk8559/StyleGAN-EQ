import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from utility.attributes import generate_features


def clip_top(s, top=1.0):
    return np.clip(s.copy(), 0, top)


def norm_post_synthesis(s):
    s = (s + 1) * 0.5
    s = np.clip(s, 0, 10)
    return s


def mapping_batch(n, z, G, b=32):
    w = []
    bn = n//b + ((n%b)>0)
    print('Mapping {} batches...'.format(bn))
    for i in tqdm(range(bn)):
        iend = np.minimum((i+1)*b, n)
        w.append(G.mapping(z[i*b:iend], None))
    w = torch.cat(w, 0)
    assert w.shape[0]==n
    return w


def synthesis_batch(n, w, G, b=32, log=True):
    s = []
    bn = n//b + ((n%b)>0)
    if log:
        print('Synthesis {} batches...'.format(bn))
    for i in tqdm(range(bn)):
        iend = np.minimum((i+1)*b, n)
        s.append(G.synthesis(w[i*b:iend], noise_mode='const'))
    s = torch.cat(s)
    assert s.shape[0]==n
    return s


def random_w_features(num, Gpath, bs=32, align_l0=False):
    with open(Gpath, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    
    z = torch.randn([num, G.z_dim]).cuda()
    w = mapping_batch(num, z, G, b=bs)
    
    if align_l0:
        w_syn = w.clone().detach()
        w_syn[:, 0, :] = torch.mean(w_syn[:, 0, :], 0, keepdim=True)
        s = synthesis_batch(num, w_syn, G, b=bs)
    else:
        s = synthesis_batch(num, w, G, b=bs)
    
    w = w.cpu().numpy()[:, 0, :]
    s = s.cpu().numpy().squeeze()
    s = norm_post_synthesis(s)
    
    features = [generate_features(s[n]) for n in range(num)]
    features = np.stack(features, 0).astype(np.float32)
    return w, features


# for w-f datasets
def next_fid(fdir):
    flist =  os.listdir(fdir)
    fid = 0
    for name in flist:
        fid = np.maximum(fid, int(name[2:name.find('.npy')]) + 1)
    return fid


# for w-f datasets
def load_w_features(fdir):
    x, y = [], []
    flist =  os.listdir(fdir)
    for f in flist:
        if f[0]=='w':
            x.append(np.load(fdir + f))
        if f[0]=='f':
            y.append(np.load(fdir + f))
    x = np.concatenate(x, 0)
    y = np.concatenate(y, 0)
    return x, y
