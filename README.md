# StyleGAN-EQ
**High-resolution ground motion generation with time-frequency representation**<br>
*Zekun Xu, Jun Chen(CA)*

>**Abstract:** Data-driven deep learning application in earthquake engineering highlights the insufficient quantity and the imbalanced feature distribution of measured ground motions, which can be mitigated with artificial ones. Traditional ground motion generation techniques tend to extend the catalogs conditioning on existing records, while current deep learning-based methods such as generative adversarial networks (GANs) only provide limited duration or sampling rate, thus obstructing further applications. In this paper, an invertible time-frequency transformation process is employed, based on which the transformed earthquake representation is implicitly modeled by advanced GANs for high-resolution and unconditional generation. Moreover, leveraging the disentangling property of the GAN's latent space, the newly developed latent space walking method is adopted to assure the generations with controllable time-frequency features. A feature-balanced generated ground motion dataset has been constructed in combination with the proposed methods, and the application potential was demonstrated through comparative experiments of different datasets.

## Ground Motion Datasets
Records are saved as spectrograms in `*.png` format at `datasets`, use **GLA** for time histories.
| Filename | Samples | Source | Site Class |
| :--- | :--- | :--- | :--- |
| `kiknet-15k-128x128.zip` | 15,780 | KiK-net | C |
| `kiknet-fake-part1-128x128.zip` | 32,000 | `stylegan3-kiknet-15k.pkl` | - |
| `kiknet-fake-part2-128x128.zip` | 54,400 | `stylegan3-kiknet-15k.pkl` | - |
| `peer-7k-classC-128x128.zip` | 3,798 | PEER NGA-West2 | C |
| `peer-7k-classCD-128x128.zip` | 7,174 | PEER NGA-West2 | C&D |

## Pretrained Models
Models can be downloaded with links at `pretrained_models`.
| Filename | Type | Capacity | Training set |
| :--- | :--- | :--- | :--- |
| `stylegan3-kiknet-15k.pkl` | stylegan3-T, unconditional | 8,192 | `kiknet-15k` |
| `stylegan3-peer-7k-classC.pkl` | stylegan3-T, unconditional | 8,192 | `peer-7k-classC` |
| `stylegan3-peer-7k-classCD.pkl` | stylegan3-T, unconditional | 8,192 | `peer-7k-classCD` |

## Dependencies
Refer to StyleGAN3 (https://github.com/NVlabs/stylegan3)

## Citation
If you use this data or code for your research, please cite our paper:

```
Zekun Xu, Jun Chen,
High-resolution ground motion generation with time-frequency representation,
Bulletin of Earthquake Engineering,
2024,
https://doi.org/10.1007/s10518-024-01912-1.
(https://link.springer.com/article/10.1007/s10518-024-01912-1)
```
