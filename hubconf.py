import torch
import pickle
import dnnlib
import PIL.Image
import numpy as np
from typing import Tuple


def stylegan3_EQ(pretrained='kiknet-15k'):
    model_dic = {
        'kiknet-15k': 'https://drive.google.com/file/d/1xREaKxQMDta5U0371NYZb9cuMDQs87bL/view?usp=sharing',
        'peer-7k-classC': 'https://drive.google.com/file/d/13g3pdQuadU6nQBTOqBl2XTeKzuJ-kiVM/view?usp=sharing',
        'peer-7k-classCD': 'https://drive.google.com/file/d/17hty6uf6kKiURlfur76xjtq-AVN85vnd/view?usp=sharing'
    }
    
    url = model_dic.get(pretrained, None)
    if url is  None:
        raise ValueError(f'Model {pretrained} not found')
    return gen_uncond_pipeline(url)


def make_transform(translate, angle):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


class gen_uncond_pipeline:
    def __init__(self, url, device:str=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        with dnnlib.util.open_url(url) as f:
            self.G = pickle.load(f)['G_ema'].to(self.device)

    def __call__(self, seed:int=2021, translate:Tuple[int]=(0, 0), rotate:float=0.0,):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device)

        if hasattr(self.G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            self.G.synthesis.input.transform.copy_(torch.from_numpy(m))

        img = self.G(z, None, truncation_psi=1., noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        return img
