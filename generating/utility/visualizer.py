import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_W_space(
        w, f,
        perplexity = 25,
        pct_f      = [2.5, 97.5],
        pct_emb    = [0.1, 99.9],
        f_plot     = 0,
        load       = False,
        path       = None
        ):
    '''
    
    Parameters
    ----------
    w : Array(n, 512)
        Ramdomly sampled data in W space.
    f: Array(n, 5)
        Labels of spectrograms corresponding to {w}.
    perplexity: int
        TSNE parameter.
    pct_f : list[2,], optional
        Percentiles of labels to cut-off.
        The default is [2.5, 97.5].
    pct_emb : list[2,], optional
        Percentiles of tsne embeddings to cut-off.
        The default is [0.1, 99.9].
    f_plot : int, optional
        Determines which feature to plot.
        Labels: [energy, f_max, t_max, f_range, t_range]
        The default is 0 (energy).
    load : bool, optional
        Determines whether to create new embeddings or load from files.
        New embeddings will overwrite old files.
        The default is False.

    Return
    -------
    None.

    '''
    if load:
        emb3 = np.load(path + '/tsne_embedding.npy')
        f3   = np.load(path + '/tsne_feature.npy')
    else:
        tsne = TSNE(2,
                    perplexity         = perplexity,
                    early_exaggeration = 12,
                    learning_rate      = 100,
                    init               = 'pca',
                    verbose            = 1)
        emb = tsne.fit_transform(w)
        
        f2 = []
        emb2 = []
        pct1 = np.percentile(f, pct_f[0], axis=0)
        pct2 = np.percentile(f, pct_f[1], axis=0)
        for i in range(emb.shape[0]):
            if (f[i][f_plot] > pct1[f_plot]):
                if (f[i][f_plot] < pct2[f_plot]).all():
                    f2.append(f[i])
                    emb2.append(emb[i])
        f2 = np.stack(f2, 0)
        emb2 = np.stack(emb2, 0)
        
        f3 = []
        emb3 = []
        pct1 = np.percentile(emb2, pct_emb[0], axis=0)
        pct2 = np.percentile(emb2, pct_emb[1], axis=0)
        for i in range(emb2.shape[0]):
            if (emb2[i][0] > pct1[0]) and (emb2[i][1] > pct1[1]):
                if (emb2[i][0] < pct2[0]) and (emb2[i][1] < pct2[1]):
                    f3.append(f2[i])
                    emb3.append(emb2[i])
        f3 = np.stack(f3, 0)
        emb3 = np.stack(emb3, 0)
    
    plt.figure(figsize=(20,20))
    plt.scatter(emb3[:, 0], emb3[:, 1], c=f3[:, f_plot])
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    np.save(path + '/tsne_embedding.npy', emb3)
    np.save(path + '/tsne_feature.npy', f3)
    return