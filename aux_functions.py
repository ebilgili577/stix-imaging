import os
import numpy as np
from scipy.io import readsav
import matplotlib.pyplot as plt


def Fourier_matrix_STIX(u, v, n_pix, pix_size):
    """
    Function for creating the Fourier matrix to be used for computing the STIX visibility values from an image
    
    INPUTS:
    
    u: numpy array
        float array containing the u coordinates of the frequencies sampled by STIX
    
    v: numpy array
        float array containing the v coordinates of the frequencies sampled by STIX
    
    n_pix: int
        number of pixels (rows/columns) used to discretize the considered Field-of-View (FOV)
    
    pix_size: float
        pixel size (in arcsec)
    """
    
    x = np.linspace(-(n_pix - 1)/2, (n_pix - 1)/2, num=n_pix)*pix_size
    x = np.expand_dims(x, axis=0)
    x = np.repeat(x, n_pix, axis=0)

    y = np.linspace((n_pix - 1)/2, -(n_pix - 1)/2, num=n_pix)*pix_size
    y = np.expand_dims(y, axis=1)
    y = np.repeat(y, n_pix, axis=1)
    
    dim = len(u)
    F = np.zeros((2*dim, n_pix*n_pix))
    
    for i in range(dim):
        
        phase = 2*np.pi*(x * u[i] + y * v[i])
        F[i, :]    = np.reshape(np.cos(phase), (n_pix*n_pix,))
        F[i+dim, :] = np.reshape(np.sin(phase), (n_pix*n_pix,))
        
    return F * pix_size**2

    
def stx_plot_vis_fit(vis, vis_pred, sigamp):

    # Compute amplitude and phase of the observed visibilities
    re_vis = vis[0:24]
    im_vis = vis[24:]
    
    amp_vis   = np.sqrt( re_vis**2 + im_vis**2 )
    phase_vis = np.arctan2(im_vis, re_vis) / np.pi * 180. # In degrees

    sigphase= np.divide(sigamp, amp_vis, out=np.zeros_like(sigamp), where=amp_vis!=0) / np.pi * 180.
    
    # Compute amplitude and phase of predict visibilities
    # vis_pred = f_matrix @ stx_map.flatten()

    re_vis_pred = vis_pred[0:24]
    im_vis_pred = vis_pred[24:]
    
    amp_vis_pred   = np.sqrt( re_vis_pred**2 + im_vis_pred**2 )
    phase_vis_pred = np.arctan2(im_vis_pred, re_vis_pred) / np.pi * 180. # In degrees

    # Compute chi2
    diff_re = re_vis - re_vis_pred
    diff_im = im_vis - im_vis_pred
    chi2 = np.sum((diff_re**2 + diff_im**2)/sigamp**2) / 23
    
    # Makeplot
    
    idx_10 = [0,1,2]
    idx_9  = [3,4,5]
    idx_8  = [6,7,8]
    idx_7  = [9,10,11]
    idx_6  = [12,13,14]
    idx_5  = [15,16,17]
    idx_4  = [18,19,20]
    idx_3  = [21,22,23]

    idx_vis = idx_3+idx_4+idx_5+idx_6+idx_7+idx_8+idx_9+idx_10
    
    xx = np.linspace(0, 29, num=30) / 3 + 1.2
    xx = xx[6:]
    
    markersize = 8
    fontsize = 15
    ticksize = 12
    
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(15, 7), gridspec_kw={'height_ratios': [2, 1]})
    fig.subplots_adjust(hspace=0)

    #-------------- PHASES
    axs[0,0].plot(xx, xx*0, '--', color='grey')
    axs[0,0].errorbar(xx, phase_vis[idx_vis], yerr=sigphase[idx_vis], \
                      fmt='s', color='blue',ecolor='green', mfc='none', markersize=markersize,\
                      label='Observed')
    axs[0,0].plot(xx, phase_vis_pred[idx_vis], 'ro', mfc='none', markersize=markersize, label='Predicted')

    for i in range(7):
        axs[0,0].plot([i+4,i+4],[-200,200], ':', color='grey')
    
    axs[0,0].set_ylim([-200,200])
    axs[0,0].set_xlim([3,11])
    axs[0,0].set_xticks([3,4,5,6,7,8,9,10])#, labels=["3","4","5","6","7","8","9","10"])
    axs[0,0].set_xticklabels(["3","4","5","6","7","8","9","10"])
    axs[0,0].tick_params(axis='both', which='major', labelsize=ticksize)
    axs[0,0].set_title('VISIBILITIES PHASE PLOT',fontsize=fontsize)
    axs[0,0].legend(fontsize=fontsize)
    axs[0,0].set_ylabel('[deg]',fontsize=fontsize)

    axs[1,0].plot(xx, xx*0, '--', color='grey')
    axs[1,0].plot(xx, xx*0+3, ':', color='grey')
    axs[1,0].plot(xx, xx*0-3, ':', color='grey')
    axs[1,0].set_ylim([-8,8])
    for i in range(7):
        axs[1,0].plot([i+4,i+4],[-8,8], ':', color='grey')

    axs[1,0].plot(xx, (phase_vis[idx_vis]-phase_vis_pred[idx_vis])/sigphase[idx_vis], '+', color='k', \
                  markersize=markersize)
    axs[1,0].set_ylabel('Residuals',fontsize=fontsize)
    axs[1,0].tick_params(axis='both', which='major', labelsize=ticksize)
    axs[1,0].set_xlabel('Detector label',fontsize=fontsize)

    #-------------- AMPLITUDES

    max_amp = max([np.max(amp_vis), np.max(amp_vis_pred)])
    
    axs[0,1].errorbar(xx, amp_vis[idx_vis], yerr=sigamp[idx_vis], \
                      fmt='s', color='blue',ecolor='green', mfc='none', markersize=markersize,\
                      label='Observed')
    axs[0,1].plot(xx, amp_vis_pred[idx_vis], 'ro', mfc='none', markersize=markersize, label='Predicted')

    for i in range(7):
        axs[0,1].plot([i+4,i+4],[0,max_amp*1.2], ':', color='grey')
    
    axs[0,1].set_ylim([0,max_amp*1.2])
    axs[0,1].set_xlim([3,11])
#     axs[0,1].set_xticks([3,4,5,6,7,8,9,10], labels=["3","4","5","6","7","8","9","10"])
    axs[0,1].set_xticks([3,4,5,6,7,8,9,10])#, labels=["3","4","5","6","7","8","9","10"])
    axs[0,1].set_xticklabels(["3","4","5","6","7","8","9","10"])
    axs[0,1].tick_params(axis='both', which='major', labelsize=ticksize)
    axs[0,1].set_title('VISIBILITIES AMPLITUDES PLOT',fontsize=fontsize)
    axs[0,1].legend(fontsize=fontsize)
    axs[0,1].set_ylabel('[counts s$^{-1}$ cm$^{-2}$ keV$^{-1}$]',fontsize=fontsize)

    axs[1,1].plot(xx, xx*0, '--', color='grey')
    axs[1,1].plot(xx, xx*0+3, ':', color='grey')
    axs[1,1].plot(xx, xx*0-3, ':', color='grey')
    axs[1,1].set_ylim([-8,8])
    for i in range(7):
        axs[1,1].plot([i+4,i+4],[-8,8], ':', color='grey')

    axs[1,1].plot(xx, (amp_vis[idx_vis]-amp_vis_pred[idx_vis])/sigamp[idx_vis], '+', color='k', \
                  markersize=markersize)
    axs[1,1].set_ylabel('Residuals',fontsize=fontsize)
    axs[1,1].tick_params(axis='both', which='major', labelsize=ticksize)
    axs[1,1].set_xlabel('Detector label',fontsize=fontsize)
    
    plt.suptitle("CHI2 :%9.2f"%chi2, fontsize=20)
    plt.show()

def compute_chi2(vis, vis_pred, sigamp):

    # Compute amplitude and phase of the observed visibilities
    re_vis = vis[0:24]
    im_vis = vis[24:]
    
    re_vis_pred = vis_pred[0:24]
    im_vis_pred = vis_pred[24:]

    # Compute chi2
    diff_re = re_vis - re_vis_pred
    diff_im = im_vis - im_vis_pred
    chi2 = np.sum((diff_re**2 + diff_im**2)/sigamp**2) / 23

    return chi2