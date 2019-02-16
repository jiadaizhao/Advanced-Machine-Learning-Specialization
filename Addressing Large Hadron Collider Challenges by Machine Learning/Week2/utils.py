import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn.metrics import roc_curve, roc_auc_score


label_class_correspondence = {'Electron': 0, 'Ghost': 1, 'Kaon': 2, 'Muon': 3, 'Pion': 4, 'Proton': 5}
class_label_correspondence = {0: 'Electron', 1: 'Ghost', 2: 'Kaon', 3: 'Muon', 4: 'Pion', 5: 'Proton'}


def get_class_ids(labels):
    """
    Convert particle type names into class ids.

    Parameters:
    -----------
    labels : array_like
        Array of particle type names ['Electron', 'Muon', ...].

    Return:
    -------
    class ids : array_like
        Array of class ids [1, 0, 3, ...].
    """
    return numpy.array([label_class_correspondence[alabel] for alabel in labels])


def plot_roc_curves(predictions, labels):
    """
    Plot ROC curves.

    Parameters:
    -----------
    predictions : array_like
        Array of particle type predictions with shape=(n_particles, n_types).
    labels : array_like
        Array of class ids [1, 0, 3, ...].
    """
    plt.figure(figsize=(9, 6))
    u_labels = numpy.unique(labels)
    for lab in u_labels:
        y_true = labels == lab
        y_pred = predictions[:, lab]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        plt.plot(tpr, 1-fpr, linewidth=3, label=class_label_correspondence[lab] + ', AUC = ' + str(numpy.round(auc, 4)))
        plt.xlabel('Signal efficiency (TPR)', size=15)
        plt.ylabel("Background rejection (1 - FPR)", size=15)
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.xlim(0., 1)
        plt.ylim(0., 1)
        plt.legend(loc='lower left', fontsize=15)
        plt.title('One particle vs rest ROC curves', loc='right', size=15)
        plt.grid(b=1)
        
        
def my_percentile(arr, w, q):

    left = 0.
    right = (w).sum()
    sort_inds = numpy.argsort(arr, axis=0)
    if left/right >= q/100.:
        return arr[0]
    for i in sort_inds:
        left += w[i]
        if left/right >= q/100.:
            return arr[i]

def base_plot(prediction, spectator, cut, percentile=True, weights=None, n_bins=100,
              color='b', marker='o', ms=4, label="MVA", fmt='o', markeredgecolor='b', markeredgewidth=2, ecolor='b'):
    """
    Base plot for signal efficiency.

    Parameters:
    -----------
    prediction : array_like
        Array of predictions for signal for a selected particle type with shape=(n_particles, ).
    spectator : array_like
        To plot dependence of signal efficiency on this feature.
    cut : float
        Global efficiency value.
    bins : int
        Number of bin for plot.
    """
    if weights is None:
        weights = numpy.ones(len(prediction))

    if percentile:
        if weights is None:
            cut = numpy.percentile(prediction, 100-cut)
        else:
            cut = my_percentile(prediction, weights, 100-cut)
    
    edges = numpy.linspace(spectator.min(), spectator.max(), n_bins)
    
    xx = []
    yy = []
    xx_err = []
    yy_err = []
    
    for i_edge in range(len(edges)-1):

        left = edges[i_edge]
        right = edges[i_edge + 1]
        
        N_tot_bin = weights[((spectator >= left) * (spectator < right))].sum()
        N_cut_bin = weights[((spectator >= left) * (spectator < right) * (prediction >= cut))].sum()
        
        if N_tot_bin != 0:
            
            x = 0.5 * (right + left)
            y = 1. * N_cut_bin / N_tot_bin
            
            if y > 1.:
                y = 1.
            if y < 0:
                y = 0
            
            xx.append(x)
            yy.append(y)
            
            x_err = 0.5 * (right - left)
            y_err = numpy.sqrt(y*(1-y)/N_tot_bin)
            
            xx_err.append(x_err)
            yy_err.append(y_err)
        
        else:
            pass

    plt.errorbar(xx, yy, yerr=yy_err, xerr=xx_err, fmt=fmt, color=color, marker=marker, ms=ms, label=label, markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth, ecolor=ecolor)
    
    return cut

def plot_signal_efficiency(predictions, labels, spectator, eff=60, n_bins=20, xlabel='Spectator'):
    """
    Plot dependence of signal efficiency from spectator feature for all particle types.

    Parameters:
    -----------
    prediction : array_like
        Array of predictions for signal for a selected particle type with shape=(n_particles, ).
    labels : array_like
        Array of class ids [1, 0, 3, ...].
    spectator : array_like
        To plot dependence of signal efficiency on this feature.
    cut : float
        Global efficiency value.
    bins : int
        Number of bin for plot.
    xlabel : string
        Label of x-axis.
    """
    
    plt.figure(figsize=(5.5*2, 3.5*3))
    u_labels = numpy.unique(labels)
    for lab in u_labels:
        y_true = labels == lab
        pred = predictions[y_true, lab]
        spec = spectator[y_true]
        plt.subplot(3, 2, lab+1)
        base_plot(pred, spec, cut=eff, percentile=True, weights=None, n_bins=n_bins, color='1', marker='o', 
                  ms=7, label=class_label_correspondence[lab], fmt='o')
        
        plt.plot([spec.min(), spec.max()], [eff / 100., eff / 100.], label='Global signal efficiecny', color='r', linewidth=3)
        plt.legend(loc='best', fontsize=12)
        plt.xticks(size=12)
        plt.yticks(size=12)
        plt.ylabel('Signal efficiency (TPR)', size=12)
        plt.xlabel(xlabel,size=12)
        plt.ylim(0, 1)
        plt.xlim(spec.min(), spec.max())
        plt.grid(b=1)
    plt.tight_layout()
        

def plot_signal_efficiency_on_p(predictions, labels, spectator, eff=60, n_bins=20):
    """
    Plot dependence of signal efficiency from particle momentum feature for all particle types.

    Parameters:
    -----------
    prediction : array_like
        Array of predictions for signal for a selected particle type with shape=(n_particles, ).
    labels : array_like
        Array of class ids [1, 0, 3, ...].
    spectator : array_like
        To plot dependence of signal efficiency on this feature.
    cut : float
        Global efficiency value.
    bins : int
        Number of bin for plot.
    """
    sel = spectator < 200 * 10**3
    plot_signal_efficiency(predictions[sel], labels[sel], spectator[sel] / 10**3, eff, n_bins, 'Momentum, GeV/c')
    

def plot_signal_efficiency_on_pt(predictions, labels, spectator, eff=60, n_bins=20):
    """
    Plot dependence of signal efficiency from particle transverse momentum feature for all particle types.

    Parameters:
    -----------
    prediction : array_like
        Array of predictions for signal for a selected particle type with shape=(n_particles, ).
    labels : array_like
        Array of class ids [1, 0, 3, ...].
    spectator : array_like
        To plot dependence of signal efficiency on this feature.
    cut : float
        Global efficiency value.
    bins : int
        Number of bin for plot.
    """
    sel = spectator < 10 * 10**3
    plot_signal_efficiency(predictions[sel], labels[sel], spectator[sel] / 10**3, eff, n_bins, 'Transverse momentum, GeV/c')
    
    
    
from IPython.display import FileLink       
def create_solution(ids, proba, filename='submission_file.csv.zip'):
    """saves predictions to file and provides a link for downloading """
    solution = pandas.DataFrame({'ID': ids})
    for name in ['Ghost', 'Electron', 'Muon', 'Pion', 'Kaon', 'Proton']:
        solution[name] = proba[:, label_class_correspondence[name]]
    solution.to_csv('{}'.format(filename), index=False, float_format='%.5f', compression="gzip")
    return FileLink('{}'.format(filename))