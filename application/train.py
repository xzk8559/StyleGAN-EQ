import os
import joblib
import argparse
import numpy as np
from PIL import Image
from sklearn.preprocessing import RobustScaler
# import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Response spectra prediction in Section Application')
    
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--nofake', action='store_true', default=False)
    parser.add_argument('--noreal', action='store_true', default=False)
    
    parser.add_argument('--real_n', type=int, default=15780)
    parser.add_argument('--fake_n', type=int, default=32000)
    parser.add_argument('--train_n', type=int, default=47780)
    
    parser.add_argument('--model', type=str, default='simcnn')
    parser.add_argument('--interpolate', action='store_true', default=False)
    parser.add_argument('--root', type=str, default='../../data/')
    
    return parser.parse_args()


args = parse_args()

# -------------------------------path config-----------------------------------

np.random.seed(2021)
train_real_i = list(np.sort(np.random.choice(15780, args.real_n, replace=False)))
np.random.seed(2021)
train_fake_i = list(np.sort(np.random.choice(32000, args.fake_n, replace=False)))

real_root = args.root + 'kiknet/'
fake_root = args.root + 'kiknet-fake/balanced/'

real_spec_fname = sorted(os.listdir(real_root + 'spec128/images/'))
fake_spec_fname = sorted(os.listdir(fake_root + 'spec128/images/'))

real_paths = []
real_paths.append(real_root + 'rspectra/0.05-2.00/rspec.npy')
real_paths.append(real_root + 'spec128/images/')
real_paths.append(real_root + 'spec128/scale.npy')
real_paths.append(real_spec_fname)

fake_paths = []
fake_paths.append(fake_root + 'rspectra/0.05-2.00/rspec.npy')
fake_paths.append(fake_root + 'spec128/images/')
fake_paths.append(fake_root + 'spec128/scale.npy')
fake_paths.append(fake_spec_fname)

# --------------------------------load data------------------------------------

def load_xy(paths: list, inds: list):
    sa_p, spec_p, scale_p, spec_fn = paths
    sa = np.load(sa_p)[inds]
    scale = np.load(scale_p)[inds]
    fpath = [spec_p + spec_fn[i] for i in inds]
    return sa, scale.reshape((-1,)), fpath

# train-real / train-fake
train_sa1, train_scale1, train_fpath1 = load_xy(real_paths, train_real_i)
train_sa2, train_scale2, train_fpath2 = load_xy(fake_paths, train_fake_i)

if args.nofake:
    train_sa = train_sa1
    train_fpath = train_fpath1
    train_scale = train_scale1
    args.fake_n = 0
elif args.noreal:
    train_sa = train_sa2
    train_fpath = train_fpath2
    train_scale = train_scale2
    args.real_n = 0
else:
    train_fpath = train_fpath1 + train_fpath2
    train_sa = np.concatenate((train_sa1, train_sa2), 0)
    train_scale = np.concatenate((train_scale1, train_scale2), 0)

scaler = RobustScaler(quantile_range=(5.0, 95.0)).fit(train_sa.reshape((-1, 3)))
dset_scale = np.percentile(train_scale, 95)
joblib.dump(scaler, 'save/scaler.save')

def split_tra_val():
    np.random.seed(2021)
    all_ind = list(np.arange(len(train_fpath)))
    val_ind = list(np.sort(np.random.choice(len(train_fpath), len(train_fpath)//10, replace=False)))
    tra_ind = list(set(all_ind).difference(set(val_ind)))
    tra_fpath = np.array(train_fpath)[tra_ind]
    val_fpath = np.array(train_fpath)[val_ind]
    tra_sa = train_sa[tra_ind]
    val_sa = train_sa[val_ind]
    tra_scale = train_scale[tra_ind]
    val_scale = train_scale[val_ind]
    return [tra_sa, tra_scale, tra_fpath], [val_sa, val_scale, val_fpath]

tmp1, tmp2 = split_tra_val()
train_sa, train_scale, train_fpath = tmp1
val_sa, val_scale, val_fpath = tmp2

print('train number {}'.format(len(train_sa)))
print('valid number {}'.format(len(val_sa)))
args.train_n = len(train_sa)

del tmp1, tmp2
del train_real_i, train_fake_i
del train_fpath1, train_fpath2, train_sa1, train_sa2, train_scale1, train_scale2
del real_root, real_paths, real_spec_fname
del fake_root, fake_paths, fake_spec_fname

# ----------------------------------train--------------------------------------

def random_offset(s):
    tmp = np.argwhere(s>0.05)[:, 1]
    left = np.min(tmp)
    right = s.shape[1] - np.max(tmp)
    offset = np.random.randint(1 + left + right)
    s_full = np.concatenate((np.zeros((128,right)), s, np.zeros((128,left))), -1)
    s = s_full[:, offset:offset+128]
    return s

class specDataset(Dataset):
    def __init__(self, fpath, sp, scl, train: bool):
        self.scl = scl
        self.fpath = fpath
        self.train = train
        self.sp = sp.reshape((-1, 40, 3))
        self.len = len(fpath)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        s = Image.open(self.fpath[idx])
        s = np.array(s) / 255.
        
        if self.train:
            s = random_offset(s)
        s = s.reshape(1, 128, 128) * self.scl[idx] / dset_scale
        s = torch.Tensor(s.astype('float32'))
        
        sp = self.sp[idx].reshape((40, 3))
        sp = scaler.transform(sp)
        sp = torch.Tensor(sp.reshape((-1,)).astype('float32'))
        return s, sp
    
    
def run(args):
    dloader = DataLoader(
        specDataset(train_fpath, train_sa, train_scale, train=True),
        batch_size=args.bs, shuffle=True, pin_memory=True, num_workers=4)
    dloader_val = DataLoader(
        specDataset(val_fpath, val_sa, val_scale, train=False),
        batch_size=args.bs, shuffle=False, pin_memory=True, num_workers=4)
    
    if args.model=='simcnn':
        from model.simcnn import simCNN
        net = simCNN().cuda()
        args.interpolate = False
    elif args.model=='resnet':
        from model.resnet18 import resnet18
        net = resnet18().cuda()
        args.interpolate = True
    elif args.model=='regnet':
        from model.regnet import RegNetY_200mf
        net = RegNetY_200mf().cuda()
        args.interpolate = True
    
    print(net)
    print("{} paramerters in total".format(sum(c.numel() for c in net.parameters())))
    
    criterion = torch.nn.L1Loss().cuda()
    # criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    
    vloss_min = 1e+8
    iter_best = 0
    for t in range(args.epochs):
        net.train()
        x, ytrue = next(iter(dloader))
        if args.interpolate: x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        y = net(x.cuda())
    
        loss = criterion(y, ytrue.cuda())
        # wandb.log({"loss": loss}, step=t)
        if t % 10 == 9:
            vloss, _ = val(net, dloader_val, criterion, t)
            print(t+1, loss.item(), vloss)
            
            if vloss_min>vloss:
                vloss_min = vloss
                iter_best = t
                torch.save(net.state_dict(), 'save/best.pkl')
                # wandb.run.summary["best-vloss"] = vloss_min
                # wandb.run.summary["best-epoch"] = iter_best
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    torch.save(net.state_dict(), 'save/final.pkl')
    print('Best loss: {:.9f} at iteration {}'.format(vloss_min, iter_best))
    return net


def val(net, dataloader_val, criterion, step):
    net.eval()
    vloss = []
    pred = []
    with torch.no_grad():
        for ii, data in enumerate(dataloader_val):
            x, ytrue = data
            if args.interpolate: x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
            ypred = net(x.cuda())
            vloss0 = criterion(ypred, ytrue.cuda())
            vloss.append(vloss0.cpu())
            pred.append(ypred.detach().cpu().numpy().reshape((-1, 40, 3)))
    vloss = np.mean(vloss)
    # wandb.log({"vloss": np.mean(vloss)}, step=step)
    pred = np.concatenate(pred, 0)
    pred = scaler.inverse_transform(pred.reshape((-1, 3))).reshape((-1, 40, 3))
    
    mmape = 0
    # log_name = ['response-acc', 'response-vel', 'response-dis']
    # log_dict = dict()
    for i in range(3):
        # mae = np.mean(np.abs(val_sa[:,:,i]-pred[:,:,i]))
        # mse = np.mean(np.abs(val_sa[:,:,i]-pred[:,:,i])**2)
        mape = np.mean(np.abs(pred[:,:,i]/val_sa[:,:,i]-1))
        # log_dict[log_name[i] + '-mae'] = mae
        # log_dict[log_name[i] + '-mse'] = mse
        # log_dict[log_name[i] + '-mape'] = mape
        mmape += mape / 3.
    # log_dict['mean-mape'] = mmape
    # wandb.log(log_dict, step=step)
    net.train()
    return vloss, mmape


# wandb.init(project='stylegan/application')
net = run(args)
