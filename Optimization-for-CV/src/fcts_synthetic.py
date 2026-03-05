import numpy as np 
import pandas as pd 
from skimage import exposure, img_as_float, color
import pywt
import cv2
from PIL import Image
import scipy
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

def edge_keeping_index(original_image, filtered_image):
    # Compute Laplacian of the original and filtered images
    laplacian_original = (cv2.Laplacian((original_image*255).astype(np.uint8), cv2.CV_64F)).reshape(original_image.shape[0:2])
    laplacian_filtered = (cv2.Laplacian((filtered_image*255).astype(np.uint8), cv2.CV_64F)).reshape(filtered_image.shape[0:2])


    # Compute the edge-keeping index
    num = np.sum(np.dot((laplacian_original - np.mean(laplacian_original)).T, (laplacian_filtered - np.mean(laplacian_filtered))))
    denom = np.sum(np.dot((laplacian_original - np.mean(laplacian_original)).T, (laplacian_original - np.mean(laplacian_original))))*np.sum(np.dot((laplacian_filtered - np.mean(laplacian_filtered)).T, (laplacian_filtered - np.mean(laplacian_filtered))))
    eki = num /np.sqrt(denom)
    return eki

def Hard_thresholding(t): 
    global H
    res = np.zeros_like(H)
    for i in range(H.shape[0]): 
        for j in range(H.shape[1]):
            if np.abs(H[i,j])>=t: 
                res[i,j] = H[i,j]
    return res

def Soft_thresholding(t):
    global H 
    res = np.zeros_like(H)
    for i in range(H.shape[0]): 
        for j in range(H.shape[1]):
            if np.abs(H[i,j])>=t: 
                res[i,j] = np.sign(H[i,j])*(np.abs(H[i,j])-t)
    return res

def XT(tau):
    global H, nl
    res_wav = H.copy()
    for i in range(H.shape[0]): 
        for j in range(H.shape[1]): 
            if H[i,j]<tau: 
                res_wav[i,j] = H[i, j]*np.exp(nl * (np.abs(H[i, j]) - tau))
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

def all_train_synthetic(path_img, og_img, path_df, wav, methods_thr, thr): 
    global columns, img, log_img
    img = plt.imread(path_img)
    or_img = plt.imread(og_img)
    log_img = exposure.adjust_log(img_as_float(img))
    A1, (H1, V1, D1) = pywt.dwt2(log_img, wav) 
    A2, (H2, V2, D2) = pywt.dwt2(A1, wav)

    level1 = [A1, H1, D1, V1]
    level2 = [A2, H2, D2, V2]

    fig, ax = plt.subplots(nrows=len(methods_thr), ncols=len(thr), figsize=(10,10))
    fig.suptitle('Results for '+wav) 
    df = []
    for i, m in enumerate(methods_thr): 
        for j, t in enumerate(thr): 
            final_img = train_one([level1,level2], img, t, m, wav)
            df.append({'img_name':path_img.split('/')[-1].split('.')[0], 'Threshold_type':str(t).split(' ')[1], 'Threshold_function':str(m).split(' ')[1], 'EKI':edge_keeping_index(or_img, final_img), 'SSIM': structural_similarity(or_img, final_img, win_size=7,data_range=or_img.max()), 'MSE':mean_squared_error(or_img, final_img), 'PSNR':peak_signal_noise_ratio(or_img, final_img)})
            # Ajouter un titre pour chaque colonne
            if j == 0: ax[i, j].set_ylabel(str(m).split(' ')[1], rotation='vertical')
            if i == 0: ax[i, j].set_title(str(t).split(' ')[1], rotation='horizontal')
            ax[i,j].imshow(final_img, cmap='gray', interpolation='nearest')
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])  
    plt.savefig('images/synthetic/de'+path_img.split('/')[-1].split('.')[0]+wav+'.png')
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
    An1 = pywt.idwt2((tot[1][0], (leveln2[0], leveln2[1], leveln2[2])), wav)[:tot[0][0].shape[0], :tot[0][0].shape[1]]
    imglog = pywt.idwt2((An1, (leveln1[0], leveln1[1], leveln1[2])), wav)
    final_img = (exposure.rescale_intensity(np.exp(imglog[: , :img.shape[1]]), out_range=(0,1))).clip(0,1)
    return final_img 

