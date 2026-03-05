import numpy as np 
import pandas as pd 
from skimage import exposure, img_as_float, color
import pywt
import scipy
from src.utils import *
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

def Hard_thresholding(t): 
    global H
    res = H.copy()
    if t == np.nan: 
        return H
    for i in range(H.shape[0]): 
        for j in range(H.shape[1]):
            if np.abs(H[i,j, 0])<t: 
                res[i,j] = np.zeros_like(H[i,j])
    return res

def Soft_thresholding(t):
    global H 
    if t == np.nan: 
        return H
    res = np.zeros_like(H)
    for i in range(H.shape[0]): 
        for j in range(H.shape[1]):
            if np.abs(H[i,j, 0])>=t: 
                res[i,j] = np.sign(H[i,j])*(np.abs(H[i,j])-t)
    return res

def XT(tau):
    global H, nl
    res_wav = H.copy()
    if tau == np.nan:
        return res_wav
    inf_tau = np.where(np.abs(H) < tau)
    for i in inf_tau: res_wav[i] = H[i]*np.exp(nl * (np.abs(H[i]) - tau))
    return res_wav


def VisuShrink():
        global log_img, D, H
        if D.all() == 0: 
            return 0
        D_ = D[np.nonzero(D)]
        return (np.median(np.abs(D_))/0.6745)*np.sqrt(2*np.log(log_img.shape[0]*log_img.shape[1]))

def BayesShrink():
        global H, D
        D_ = D[np.nonzero(D)]
        sigma_i = (np.median(np.abs(H))/0.6745)**2 - np.std(H)**2
        sigma_x = np.sqrt(np.maximum(0, sigma_i))
        if sigma_x == 0: 
            return np.max(H)
        return (np.median(np.abs(D_))/0.6745)**2 / sigma_x

def thresholding_fct():
    global H, log_img, D, A
    if D.all == 0: 
        return 2*beta*np.abs(sigmai**2 - 0) / sigmai
    D_ = D[np.nonzero(D)]
    H_ = log_img[np.nonzero(log_img)]
    sigmai = (np.median(np.abs(H_))/0.6745)
    beta = np.sqrt(2*np.log(log_img.shape[0]*log_img.shape[1]))
    tau =  2*beta*np.abs(sigmai**2 - (np.median(np.abs(D_))/0.6745)**2) / sigmai
    return tau


methods_thr = [Soft_thresholding, Hard_thresholding, XT]
thr = [VisuShrink, BayesShrink, thresholding_fct]
columns = ['img_name', 'Threshold_type', 'Threshold_function','EKI','SSIM', 'MSE', 'PSNR']


def all_train(path_img, path_df, wav, methods_thr, thr): 
    global columns, img, log_img
    img = plt.imread(path_img)
    log_img = exposure.adjust_log(img_as_float(img))
    A1, (H1, V1, D1) = pywt.dwt2(log_img, wav) 
    A2, (H2, V2, D2) = pywt.dwt2(A1, wav)

    level1 = [A1, H1, D1, V1]
    level2 = [A2, H2, D2, V2]

    fig, ax = plt.subplots(nrows=len(methods_thr), ncols=len(thr), figsize=(10,10))
    fig.suptitle(path_img.split('/')[-1].split('.')[0]) 
    df = []
    for i, m in enumerate(methods_thr): 
        for j, t in enumerate(thr): 
            final_img = train_one([level1,level2], img, t, m, wav)
            df.append({'img_name':path_img.split('/')[-1].split('.')[0], 'Threshold_type':str(t).split(' ')[1], 'Threshold_function':str(m).split(' ')[1], 'EKI':edge_keeping_index(img, final_img), 'SSIM': structural_similarity(color.rgb2gray(img), color.rgb2gray(final_img), win_size=7,data_range=img.max()), 'MSE':mean_squared_error(img, final_img), 'PSNR':peak_signal_noise_ratio(img, final_img)})
            # Ajouter un titre pour chaque colonne
            if j == 0: ax[i, j].set_ylabel(str(m).split(' ')[1], rotation='vertical')
            if i == 0: ax[i, j].set_title(str(t).split(' ')[1], rotation='horizontal')
            ax[i,j].imshow(final_img)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])  
    plt.savefig('images/'+wav+'/9_techniques_'+wav+'/'+path_img.split('/')[-1].split('.')[0]+'.png')
    plt.tight_layout()
    plt.show()
    
    datafr = pd.DataFrame(df, columns=columns)
    if pd.read_csv(path_df, index_col=0).shape[0] != 0:
        df = pd.read_csv(path_df)[pd.Index([c for c in pd.read_csv(path_df).columns if c.split(':')[0]!='Unnamed'])]
        pd.concat([df, datafr], ignore_index=True).to_csv(path_df)
    else: datafr.to_csv(path_df, columns=columns, index=None)
    pass

def train_wavelet(path_img, path_df): 
    global img, log_img
    wavs = ['db1', 'db2', 'sym2', 'bior1.1']
    columns = ['img_name', 'wavelet','EKI','SSIM', 'MSE', 'PSNR']
    img = plt.imread(path_img)
    log_img = exposure.adjust_log(img_as_float(img))
    df = []
    fig, ax = plt.subplots(ncols=len(wavs), figsize=(10,3))
    for i, wav in enumerate(wavs): 
        A1, (H1, V1, D1) = pywt.dwt2(log_img, wav) 
        A2, (H2, V2, D2) = pywt.dwt2(A1, wav)

        level1 = [A1, H1, D1, V1]
        level2 = [A2, H2, D2, V2]

        final_img = train_one([level1,level2], img, thresholding_fct, XT, wav)
        df.append({'img_name':path_img.split('/')[-1].split('.')[0], 'wavelet':wav, 'EKI':edge_keeping_index(img, final_img), 'SSIM': structural_similarity(color.rgb2gray(img), color.rgb2gray(final_img), win_size=7,data_range=img.max()), 'MSE':mean_squared_error(img, final_img), 'PSNR':peak_signal_noise_ratio(img, final_img)})
        # Ajouter un titre pour chaque colonne
        ax[i].set_title(wav, rotation='horizontal')
        ax[i].imshow(final_img)
        ax[i].set_xticks([])
        ax[i].set_yticks([])  

    fig.suptitle('Different mother wavelets for '+path_img.split('/')[-1].split('.')[0])
    plt.savefig('images/wavelets/'+path_img.split('/')[-1].split('.')[0]+'.png')
    plt.tight_layout()
    plt.show()
    
    datafr = pd.DataFrame(df, columns=columns)
    if pd.read_csv(path_df, index_col=0).shape[0] != 0:
        df = pd.read_csv(path_df)[pd.Index([c for c in pd.read_csv(path_df).columns if c.split(':')[0]!='Unnamed'])]
        pd.concat([df, datafr], ignore_index=True).to_csv(path_df)
    else: datafr.to_csv(path_df, columns=columns, index=None)
    pass

def train_one(tot, img, thr_type, thr_fct, wav):
    global nl, D, H
    new = []
    for i,l in enumerate(tot): 
        global nl
        if i ==0: nl=1
        else: nl=0.5
        leveln = []
        for i, c in enumerate(l[1:]):
            global D, H, A
            A = l[0]
            D = l[3]
            H = c
            leveln.append(thr_fct(thr_type()))
        new.append(leveln)

    leveln1 = new[0]
    leveln2 = new[1]
    An1 = pywt.idwt2((tot[1][0], (leveln2[0], leveln2[1], leveln2[2])), wav)[:, :tot[0][0].shape[1], :3]
    imglog = pywt.idwt2((An1, (leveln1[0], leveln1[1], leveln1[2])), wav)
    final_img = exposure.rescale_intensity(np.exp(imglog[: , :img.shape[1], :3]), out_range=(0,1))
    return final_img 
