# StyleGAN-EQ
**High-resolution ground motion generation with time-frequency representation**

*Zekun Xu, Jun Chen(CA)*

Data-driven deep learning application in earthquake engineering highlights the insufficient quantity and the imbalanced feature distribution of measured ground motions, which can be mitigated with artificial ones. Traditional ground motion generation techniques tend to extend the catalogs conditioning on existing records, while current deep learning-based methods such as generative adversarial networks (GANs) only provide limited duration or sampling rate, thus obstructing further applications. In this paper, an invertible time-frequency transformation process is employed, based on which the transformed earthquake representation is implicitly modeled by advanced GANs for high-resolution and unconditional generation. Moreover, leveraging the disentangling property of the GAN's latent space, the newly developed latent space walking method is adopted to assure the generations with controllable time-frequency features. A feature-balanced generated ground motion dataset has been constructed in combination with the proposed methods, and the application potential was demonstrated through comparative experiments of different datasets.

## Dependencies
- refer to StyleGAN3 (https://github.com/NVlabs/stylegan3)
