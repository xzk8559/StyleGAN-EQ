import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as plc


'''
Attribute computation
------------------------------------------------------------------------------
'''
def energy(s):
    return np.sum(s)


def f_primary(s):
    return np.argwhere(s.max() == s)[0][0]


def t_primary(s):
    return np.argwhere(s.max() == s)[0][1]


def f_range(s, amp=0.05):
    s = s.copy()
    s[s>amp] = 1
    tmp = np.argwhere(s == 1)
    f_range = np.max(tmp[:, 0]) - np.min(tmp[:, 0])
    return f_range


def t_range(s, amp=0.05):
    s = s.copy()
    s[s>amp] = 1
    tmp = np.argwhere(s == 1)
    t_range = np.max(tmp[:, 1]) - np.min(tmp[:, 1])
    return t_range


def generate_features(s, amp=0.05):
    '''
    Parameters
    ----------
    s : Array(128, 128) or (256, 256)
        The spectrogram of an earthquake record.
        
    amp : float, optional
        The cut-off amplitude.
        Ranges are computed from pixel values > {amp}.
        Default is 0.05.

    Returns
    -------
    features: Array(5,)
        [energy, f_primary, t_primary, f_range, t_range]
    '''
    
    en = energy(s)
    fr = f_range(s, amp)
    tr = t_range(s, amp)
    fp = f_primary(s)
    tp = t_primary(s)
    return np.array([en, fp, tp, fr, tr])


# Estimate latent vector (dn)
def gradient_descent_estimator(w, lb,
                               range_lb,
                               bs=32, lr=0.003, iters=25000,
                               ini_dn=None):
    '''

    Parameters
    ----------
    w : Array(n, 512)
        Ramdomly sampled data in W space.
    lb : Array(n,)
        Labels of spectrograms corresponding to {w}.
    range_lb : List[2,]
        Ranges of labels to cut-off.
    bs : int, optional
        Batch size. The default is 32.
    lr : float, optional
        Learning rate. The default is 0.003.
    iters : int, optional
        Iterations in optimization. The default is 25000.
    ini_dn : Array(512,) or None, optional
        Determines to initial dn with a given vector or random one.
        The default is None.

    Returns
    -------
    dn_est : Array(512,)
        Estimatated attribute vector.

    '''
    
    dset = wDataset(w, lb, range_lb)
    dloader = DataLoader(dset, batch_size=bs)
    
    estimator = nn_estimator_dn(ini_dn).cuda()
    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)
    
    loss_min = 1e+8
    dn_best = 0
    iter_best = 0
    for t in range(iters):
        x, zeros = next(iter(dloader))
        y = estimator(x.cuda())
    
        loss = criterion(y, zeros.cuda())
        if t % 1000 == 999:
            print(t+1, loss.item())
            
        if loss_min>loss.item():
            dn_best = estimator.dn.clone().detach()
            loss_min = loss.item()
            iter_best = t
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    dn_est = dn_best.cpu().squeeze().numpy()
    print('Best loss: {:.9f} at iteration {}'.format(loss_min, iter_best))
    return dn_est


class wDataset(Dataset):
    def __init__(self, w, lb, rng):
        self.w = w.astype('float32')
        self.lb = lb.astype('float32')
        self.rng = rng
        self.num = w.shape[0]

    def __len__(self):
        return int(0.5 * self.num * self.num)

    def __getitem__(self, idx):
        dl = 10000
        while True:
            if np.abs(dl)>self.rng[0] and np.abs(dl)<self.rng[1]:
                break
            ij = np.random.randint(0, self.num, 2)
            dw = self.w[ij[0]] - self.w[ij[1]]
            dl = self.lb[ij[0]] - self.lb[ij[1]]
            if np.abs(dl)>0: dwl = dw / dl
        
        return dwl, np.array([0.]).astype('float32')


class nn_estimator_dn(torch.nn.Module):
    def __init__(self, ini_dn=None):
        super().__init__()
        self.dn = torch.nn.Parameter(torch.randn((512, 1)))
        if not ini_dn is None:
            dn = torch.Tensor(ini_dn.astype('float32')).unsqueeze(-1)
            self.dn = torch.nn.Parameter(dn)

    def forward(self, x):
        dn2 = (self.dn * self.dn).sum()
        return torch.mm(x, self.dn) - dn2
   
    
def average_estimator(w, lb, range_lb, p=0.1):
    num = w.shape[0]
    dn = []
    for n in tqdm(range(num)):
        for m in range(n+1, num):
            if np.random.uniform(0,1,(1,))[0]<p:
                    dl = lb[m] - lb[n]
                    if (np.abs(dl)>range_lb[0]) and (np.abs(dl)<range_lb[1]):
                        dw = w[m] - w[n]
                        dn.append( dw / dl)
    print('Computed from {} samples'.format(len(dn)))
    return np.mean(np.array(dn), 0)


def pca_estimator(w, lb, ilb):
    wlb = np.concatenate((w, lb[:, ilb:ilb+1]), 1)
    pca = PCA(n_components=1).fit(wlb)
    # print('Variance_ratio: {}'.format(pca.explained_variance_ratio_))
    
    cpn = pca.components_[0]
    dn_pca = cpn[:512] / cpn[-1]
    return dn_pca


def unit(n):
    return n / np.linalg.norm(n)


def modify(n1, n2):
    return n1 - np.dot(unit(n2), n1) * unit(n2)


def generate_dn(w, f, modified=False):
    print('Computed from {} samples'.format(w.shape[0]))
    dn = [pca_estimator(w, f, i) for i in range(f.shape[1])]
    if modified:
        dn[0] = modify(dn[0], dn[1])
        dn[1] = modify(dn[1], dn[0])
        dn[1] = modify(dn[1], dn[3])
        dn[1] = modify(dn[1], dn[4])
        dn[3] = modify(dn[3], dn[1])
        dn[3] = modify(dn[3], dn[4])
        dn[4] = modify(dn[4], dn[1])
        dn[4] = modify(dn[4], dn[3])
    return dn


def array_cosine(n1, n2):
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    return np.dot(n1, n2)


def plot_cosine_dn(dn):
    axticks = ['energy', 'f_primary', 't_primary', 'f_range', 't_range']
    cosine = []
    for i in range(len(dn)):
        cosine.append([])
        for j in range(len(dn)):
            cosine[-1].append(array_cosine(dn[i], dn[j]))
    cosine = np.array(cosine)
    
    fig, ax = plt.subplots(dpi=300)
    ax.imshow(np.abs(cosine), norm=plc.Normalize(None, 1.3))
    
    ax.set_xticks(np.arange(len(dn)))
    ax.set_yticks(np.arange(len(dn)))
    ax.set_xticklabels(axticks)
    ax.set_yticklabels(axticks)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    for i in range(len(dn)):
        for j in range(len(dn)):
            ax.text(j, i, round(cosine[i, j], 2), ha="center", va="center", color="w")
    
    ax.set_title("cosine theta of dn")
    fig.tight_layout()
    plt.show()
    return