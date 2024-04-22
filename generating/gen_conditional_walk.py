import torch
import pickle

import numpy as np
import matplotlib.pyplot as plt
import utility.attributes as attr
import utility.spectrogram as spc
import utility.visualizer as vis


'''
Generate w-(TF-feature) pairs
-----------------------------
w: Array [30000, 512]
f: Array [30000, 5  ] 
-----------------------------
The TF features are (energy, f_primary, t_primary, f_range, t_range)
'''

# model_path = '../pretrained_models/stylegan3-kiknet-15k.pkl'
# w, feature = spc.random_w_features(1000, model_path, align_l0=False)
# fdir = 'save/latent_space_walk/temp/'
# fid = spc.next_fid(fdir)
# np.save(fdir + 'w-{}.npy'.format(fid), w)
# np.save(fdir + 'f-{}.npy'.format(fid), feature)
# w, f = spc.load_w_features(fdir)

w = np.load('save/latent_space_walk/random_sample_w.npy')
f = np.load('save/latent_space_walk/random_sample_TF_feature.npy')



'''Visualize W space with t-SNE'''

save_path = 'save/latent_space_walk'
vis.visualize_W_space(w, f, perplexity=25, f_plot=0, load=1, path=save_path)
vis.visualize_W_space(w, f, perplexity=25, f_plot=1, load=1, path=save_path)
vis.visualize_W_space(w, f, perplexity=25, f_plot=2, load=1, path=save_path)
vis.visualize_W_space(w, f, perplexity=25, f_plot=3, load=1, path=save_path)
vis.visualize_W_space(w, f, perplexity=25, f_plot=4, load=1, path=save_path)



'''Estimate dn with PCA'''

dn = attr.generate_dn(w, f, modified=False)
attr.plot_cosine_dn(dn)



'''Evaluate dn with latent space walking (w/o iterating)'''

def evaluate_dn(num, Gpath, dni, f, dn, amp, interp=50, layers=None, seed=2021):
    if layers is None:
        layers = np.arange(0,16) # (1,16)
    
    with open(Gpath, 'rb') as fo:
        G = pickle.load(fo)['G_ema'].cuda()
    
    z = np.random.RandomState(seed).randn(num, G.z_dim)
    z = torch.from_numpy(z).cuda()

    w = spc.mapping_batch(num, z, G, b=32)
    w = w.cpu().numpy()
    w = np.mean(w, 0, keepdims=True)
    
    s_edit = []
    f_edit = []
    dn = dn.reshape(1,1,-1).copy()
    amps = np.arange(-interp, interp+1) / interp * amp
    for a in amps:
        w0 = w.copy()
        w0[:, layers, :] = w0[:, layers, :] + dn * a
        
        w0 = torch.from_numpy(w0).cuda()
        s0 = G.synthesis(w0, noise_mode='const')
        s0 = spc.norm_post_synthesis(s0.cpu().numpy().squeeze())
        s_edit.append(s0)
        f_edit.append(spc.generate_features(s0))
    s_edit = np.stack(s_edit, 0)
    f_edit = np.stack(f_edit, 0)
    
    plt.figure(tight_layout=True, figsize=(6,10))
    for fi in range(f_edit.shape[1]):
        p1 = np.percentile(f, 0.5, axis=0)[fi]
        p2 = np.percentile(f, 99.5, axis=0)[fi]
        p0 = p1 - 0.1 * (p2 - p1)
        p3 = p2 + 0.1 * (p2 - p1)
        p0 = np.maximum(p0, 0)
        x1 = np.percentile(f, 0.5, axis=0)[dni] - f_edit[interp, dni]
        x2 = np.percentile(f, 99.5, axis=0)[dni] - f_edit[interp, dni]
        x1 = np.maximum(x1, -amp)
        x2 = np.minimum(x2, amp)
        
        plt.subplot(f.shape[1], 1, fi+1)
        if fi==dni:
            plt.plot(amps, amps+f_edit[interp, fi], linestyle='--', color='0.8')
        else:
            plt.plot(amps, np.zeros(amps.shape)+f_edit[interp, fi], linestyle='--', color='0.8')
        plt.plot(amps, f_edit[:, fi])
        plt.axhspan(p0, p1, facecolor='0.8')
        plt.axhspan(p2, p3, facecolor='0.8')
        plt.xlim([x1, x2])
        plt.ylim([p0, p3])
        
    plt.figure(tight_layout=True, figsize=(40,4))
    inds = np.linspace(int(x1), int(x2), 10, dtype='int') + interp
    print(inds)
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(s_edit[inds[i]])
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        
    return f_edit


model_path = '../pretrained_models/stylegan3-kiknet-15k.pkl'
_ = evaluate_dn(100, model_path, 0, f, dn[0], 300, interp=300)
# _ = evaluate_dn(100, model_path, 1, f, dn[1], 80, interp=100)
# _ = evaluate_dn(100, model_path, 2, f, dn[2], 80, interp=100)
# _ = evaluate_dn(100, model_path, 3, f, dn[3], 80, interp=100)
# _ = evaluate_dn(100, model_path, 4, f, dn[4], 80, interp=100)


