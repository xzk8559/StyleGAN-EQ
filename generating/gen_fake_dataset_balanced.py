import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import utility.spectrogram as spc
import utility.attributes as attr

'''
Create a balanced fake dataset
-------------------------------------------------------------------------------
1. Generate a sample with lantent vector and spectrogram.
2. Uniformly sample the Primary Frequency from .5% to 99.5%.
3. Uniformly sample the Time Range from .5% to 99.5%.
-------------------------------------------------------------------------------
'''


w = np.load('save/latent_space_walk/random_sample_w.npy')
f = np.load('save/latent_space_walk/random_sample_TF_feature.npy')

pf, tr = f[:, 1], f[:, 4]
pf_lmt = [np.percentile(pf, .5), np.percentile(pf, 99.5)]
tr_lmt = [np.percentile(tr, .5), np.percentile(tr, 99.5)]

dn = attr.generate_dn(w, f, modified=False)
pf_dn, tr_dn = dn[1], dn[4]
del dn, f

#%% Latent space walking
import pickle
import matplotlib.pyplot as plt

def w2spec(w, G, bs=64):
    num = w.shape[0]
    w_plus = np.tile(w.reshape((num, 1, 512)), (1, 16, 1))
    s = G.synthesis(torch.Tensor(w_plus).cuda(), noise_mode='const')
    
    s = s.cpu().numpy().squeeze()
    s = spc.norm_post_synthesis(s)
    return s


def walk(w, attr, func, dn, lmt, G, num:int=10, for_vis=False):
    '''

    Parameters
    ----------
    w    : Latent space vector             | array(512,)
    dn   : Direction                       | array(512,)
    lmt  : Range of the sampled attributes | list[low, high]
    num  : Number of samples               | int
    attr : Attribute                       | int
    func : Function to compute attr        | function
    G    : StyleGAN3 model                 | module

    Returns
    -------
    w_walk    | array(num, 512)
    s_walk    | array(num, 128, 128)
    attr_walk | array(num,)
    -----------------------
    *all saved to 'data/kiknet-fake/balanced'
    
    '''
    if for_vis:
        attr_trg = np.linspace(lmt[0], lmt[1], num=num)
    else:
        attr_trg = np.sort(np.random.uniform(lmt[0], lmt[1], (num,)))
    
    w_walk = w.reshape((1, 512))
    # iteration
    for i in range(1):
        delta = (attr_trg - attr).reshape((num, 1)) * dn.reshape((1, 512))
        w_walk = w_walk + delta
        s_walk = w2spec(w_walk, G)
        attr = np.array([func(s) for s in list(s_walk)])
    
    return w_walk, s_walk
    

def walk2times(i, Gpath, num_per_walk, for_vis=False):
    with open(Gpath, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
        
    result_inter = walk(w[i], pf[i], attr.f_primary, pf_dn, pf_lmt, G, num=num_per_walk, for_vis=for_vis)

    w_walk, s_walk, attr_walk = [], [], []

    for i in (range(num_per_walk)): # tqdm
        w_i = result_inter[0][i]
        s_i = result_inter[1][i]
        tr_i = attr.t_range(s_i) # /np.max(s_i)
        
        result = walk(w_i, tr_i, attr.t_range, tr_dn, tr_lmt, G, num=num_per_walk, for_vis=for_vis)
        w_walk.append(result[0])
        s_walk.append(result[1])
        
        pf_ij = np.array([attr.f_primary(result[1][j]) for j in range(num_per_walk)])
        tr_ij = np.array([attr.t_range(result[1][j]) for j in range(num_per_walk)])
        attr_walk.append(np.stack((pf_ij, tr_ij), axis=1))
        
    w_walk = np.stack(w_walk)
    s_walk = np.stack(s_walk)
    attr_walk = np.stack(attr_walk)
    s_walk = s_walk / np.max(s_walk, (2, 3), keepdims=True)
    
    return w_walk, s_walk, attr_walk

#%% Walking test

def walking_test():
    num_per_walk = 16
    model_path = '../pretrained_models/stylegan3-kiknet-15k.pkl'
    w_walk, s_walk, attr_walk = walk2times(666, model_path, num_per_walk, for_vis=True)
    
    num_plot = np.minimum(8, num_per_walk)
    plt.figure(figsize=(20,20), dpi=300, tight_layout=True)
    for j in range(num_plot):
        for i in range(num_plot):
            plt.subplot(num_plot, num_plot, i*num_plot+j+1)
            plt.imshow(s_walk[i, j])
            plt.xticks([])
            plt.yticks([])
    del i, j, num_plot
    
    plt.figure(figsize=(8,3), dpi=300, tight_layout=True)
    plt.subplot(1, 2, 1)
    plt.imshow(attr_walk[:, :, 0])
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(attr_walk[:, :, 1])
    plt.colorbar()
    return


walking_test()

#%% Generate samples

beg, end = [5200, 5400]
num_per_walk = 4 # fixed
fdir = '../../data/kiknet-fake/balanced/'
model_path = '../pretrained_models/stylegan3-kiknet-15k.pkl'

assert end > beg
assert (len(os.listdir(fdir + 'spec128/images/'))
        + len(os.listdir(fdir + 'spec128/images_SSL/'))
        ) == beg * num_per_walk**2
print('Generating {} samples...'.format((end - beg) * num_per_walk**2))

def save_specs(s):
    flen = len(os.listdir(fdir + 'spec128/images/')) + len(os.listdir(fdir + 'spec128/images_SSL/'))
    
    for i in range(s.shape[0]):
        image = s[i].copy()
        image = image / np.max(image) * 255
        image = Image.fromarray(image.astype('uint8'))
        
        n = str(flen + i)
        for _ in range(6-len(str(flen + i))):
            n = '0' + n
            
        path = fdir + 'spec128/images/' + n + '.png'
        image.save(path)
    return


w_all, attr_all = [], []
for i in tqdm(range(beg, end)):
    w_walk, s_walk, attr_walk = walk2times(i, model_path, num_per_walk)
    w_all.append(w_walk.reshape((-1, 512)))
    attr_all.append(attr_walk.reshape((-1, 2)))
    save_specs(s_walk.reshape((-1, 128, 128)))
    
name = '{}-{}-{}.npy'.format(beg, end, num_per_walk)
np.save(fdir + 'latent_vector/' + name, np.concatenate(w_all, 0))
np.save(fdir + 'attr_pf_tr/' + name, np.concatenate(attr_all, 0))

del i, name, w_all, attr_all
del w_walk, s_walk, attr_walk

#%% After generation

def plot_balanced_attr():
    fdir = '../../data/kiknet-fake/balanced/'
    fname = os.listdir(fdir + 'attr_pf_tr/')
    attrs = np.concatenate([np.load(fdir + 'attr_pf_tr/' + fn) for fn in fname], 0)
    
    print('{} samples in total.'.format(attrs.shape[0]))
    plt.figure(figsize=(6,3), dpi=300)
    plt.subplot(1, 2, 1)
    plt.hist(attrs[:, 0], np.max(attrs[:, 0])-np.min(attrs[:, 0])+1)
    plt.subplot(1, 2, 2)
    plt.hist(attrs[:, 1], np.max(attrs[:, 1])-np.min(attrs[:, 1])+1)
    return 


plot_balanced_attr()

#%% Resample to true uniform distribution if needed

def sample1():
    import shutil
    fdir = '../../data/peerc-fake/balanced/'
    n_bins = pf_lmt[1] - pf_lmt[0] + 1
    n_specs = len(os.listdir(fdir + 'spec128/images/'))
    lmt = int(n_specs / n_bins * 0.6)
    
    lb = []
    pdf = np.zeros((128,))
    for f in os.listdir(fdir + 'spec128/images/'):
        s = Image.open(fdir + 'spec128/images/' + f)
        s = np.array(s) / 255.
        s = s / np.max(s)
        lb.append(np.array([attr.f_primary(s), attr.t_range(s)]))
        pdf[lb[-1][0]] += 1
    lb = np.stack(lb, 0)
    plt.plot(pdf)
    
    pdf = lmt / (pdf + 1e-5)
    pdf[pdf > 1] = 1
    lb_final = []
    i = 0
    for f in os.listdir(fdir + 'spec128/images/'):
        if np.random.rand() < pdf[lb[i,0]]:
            lb_final.append(lb[i])
            src = fdir + 'spec128/images/' + f
            dst = fdir + 'spec128/images_sample/' + f
            shutil.copy(src, dst)
        i += 1
    lb_final = np.stack(lb_final, 0)
    
    plt.figure()
    plt.hist(lb_final[:, 0], np.max(lb_final[:, 0])-np.min(lb_final[:, 0])+1)
    plt.figure()
    plt.hist(lb_final[:, 1], np.max(lb_final[:, 1])-np.min(lb_final[:, 1])+1)
    return


def sample2():
    import shutil
    fdir = '../../data/peerc-fake/balanced/'
    n_bins = tr_lmt[1] - tr_lmt[0] + 1
    n_specs = len(os.listdir(fdir + 'spec128/images_sample/'))
    lmt = int(n_specs / n_bins * 0.7)
    
    lb = []
    pdf = np.zeros((128,))
    for f in os.listdir(fdir + 'spec128/images_sample/'):
        s = Image.open(fdir + 'spec128/images_sample/' + f)
        s = np.array(s) / 255.
        s = s / np.max(s)
        lb.append(np.array([attr.f_primary(s), attr.t_range(s)]))
        pdf[lb[-1][1]] += 1
    lb = np.stack(lb, 0)
    plt.plot(pdf)
    
    pdf = lmt / (pdf + 1e-5)
    pdf[pdf > 1] = 1
    lb_final = []
    i = 0
    for f in os.listdir(fdir + 'spec128/images_sample/'):
        if np.random.rand() < pdf[lb[i,1]]:
            lb_final.append(lb[i])
            src = fdir + 'spec128/images_sample/' + f
            dst = fdir + 'spec128/images_sample2/' + f
            shutil.copy(src, dst)
        i += 1
    lb_final = np.stack(lb_final, 0)
    
    plt.figure()
    plt.hist(lb_final[:, 0], np.max(lb_final[:, 0])-np.min(lb_final[:, 0])+1)
    plt.figure()
    plt.hist(lb_final[:, 1], np.max(lb_final[:, 1])-np.min(lb_final[:, 1])+1)
    return


sample1()
sample2()