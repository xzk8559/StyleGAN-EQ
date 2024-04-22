import os
import torch
import joblib
import numpy as np
from PIL import Image
# import torch.nn.functional as F
from tqdm import tqdm

from model.simcnn import simCNN
# from model.resnet18 import resnet18
# from model.regnet import RegNetY_200mf


def get_scaler_real(num):
    spec_scale = {
        '50': 26.900964448360597,
        '100': 25.927047448191058,
        '200': 25.524387403134030,
        '500': 25.921620128255693,
        '1000': 25.42163201416022,
        '2000': 25.834705367400804,
        '3000': 25.961290448442977,
        '6000': 25.954351166769804,
        '12000': 26.052308058657093,
        '15780': 26.016701286292697,
        }
    scaler = joblib.load('save/scaler.save')
    return spec_scale[str(num)], scaler

def get_scaler_fake(num):
    spec_scale = {
        '50': 26.032442726408966,
        '100': 32.70959325535647,
        '200': 32.89211329218833,
        '500': 30.65527345985100,
        '1000': 31.137866704618325,
        '2000': 31.146959178648448,
        '3000': 30.892006943975172,
        '6000': 30.501065369855535,
        '12000': 30.49934377748962,
        '32000': 30.545795786460747,
        }
    scaler = joblib.load('save/scaler.save')
    return spec_scale[str(num)], scaler


dset_scale, scaler = get_scaler_real(100)
# dset_scale, scaler = get_scaler_fake(200)

test_n = 3798
test_paths = []
test_root = '../../data/peer/'
test_paths.append(test_root + 'rspectra/0.05-2.00/rspec_c.npy')
test_paths.append(test_root + 'spec128/class-c/images/')
test_paths.append(test_root + 'spec128/class-c/scale.npy')
test_paths.append( sorted(os.listdir(test_root + 'spec128/class-c/images/')) )

def load_xy(paths: list, inds: list):
    sa_p, spec_p, scale_p, spec_fn = paths
    sa = np.load(sa_p)[inds]
    scale = np.load(scale_p)[inds]
    fpath = [spec_p + spec_fn[i] for i in inds]
    return sa, scale.reshape((-1,)), fpath

test_real_i = list(np.arange(test_n))
test_rsp, test_scale, test_fpath = load_xy(test_paths, test_real_i)


net =  simCNN().cuda()
# net =  resnet18().cuda()
# net = RegNetY_200mf().cuda()
net.load_state_dict(torch.load('save/best.pkl'))

pred = []
net.eval()
with torch.no_grad():
    for i in tqdm(range(len(test_fpath))):
        s = Image.open(test_fpath[i])
        s = np.array(s) / 255.0 * test_scale[i] / dset_scale
        xtest = torch.Tensor(s.reshape((1,1,128,128))).cuda()
        # xtest = F.interpolate(xtest, size=(224, 224), mode='bilinear', align_corners=True)
        pred.append(net(xtest).detach().cpu().numpy()[0])
pred = np.array(pred).reshape((-1, 40, 3))
pred = scaler.inverse_transform(pred.reshape((-1, 3))).reshape((-1, 40, 3))
pred[pred < 0] = 0.

mae, mape = 0, 0
print('MAPE:')
for i in range(3):
    y = pred[:,:,i].flatten()
    x = test_rsp[:,:,i].flatten()

    mae += np.mean(np.abs(y-x))/3
    mape += np.mean(np.abs(y/x-1))*100/3
    print(np.mean(np.abs(y/x-1))*100)
print(mae)
print(mape)

del s, xtest
del i, x, y, mape, mae
