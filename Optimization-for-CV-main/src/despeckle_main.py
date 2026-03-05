import numpy as np 
import matplotlib.pyplot as plt 
import pywt
import scipy
from skimage import exposure, img_as_float

class Speckle_removal: 
    def __init__(self, img) -> None:
        self.img = img
        self.rows = img.shape[0]
        self.cols = img.shape[1]
        self.rgb = img.shape[2]
        self.beta = np.sqrt(2*np.log(self.rows*self.cols))
        self.img_log = exposure.adjust_log(img_as_float(img))
        pass

    def wavelet_transform(self, wav): 
        titles = ['A1',' H1','V1', 'D1', 'A2',' H2','V2', 'D2']
        A1, (H1, V1, D1) = pywt.dwt2(self.img_log, wav) 
        A2, (H2, V2, D2) = pywt.dwt2(A1, wav)

        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 10))
        k=0
        for i, a1 in enumerate([A1, H1, V1, D1, A2, H2,V2, D2]):  
            l=i 
            if i>3: 
                k=1
                l= np.mod(i, 4)
        
            ax[k,l].imshow(np.clip(a1[:, :, :1],0,1), interpolation="nearest", cmap='gray')
            ax[k,l].set_title(titles[i], fontsize=10)
            ax[k,l].set_xticks([])
            ax[k,l].set_yticks([])

        plt.tight_layout()
        plt.savefig('images/Wavelet_decomposition.png')
        plt.show()
        level1 = [A1, H1, V1, D1]
        level2 = [A2, H2, V2, D2]
        return  level1, level2
    
    def final_plot(self, final_img, method): 
        fig, ax = plt.subplots(ncols = 2)
        titles = ['Original image', 'Despeckled '+ method]
        images = [self.img, final_img]

        for i, axis in enumerate(ax): 
            axis.imshow(images[i])
            axis.set_title(titles[i], fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])

        plt.savefig('images/Denoise_results.png')
        plt.show()
        pass

    def get_despeckled_image(self, wav, method): 
        level1, level2 = self.wavelet_transform(wav)
        self._r = self.run(level1, level2, wav)
        self.final_plot(self._r, method)
        return self._r


class Paper_approach(Speckle_removal): 
    
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
    
    def XT(tau):
        global H, nl
        res_wav = H.copy()
        if tau == np.nan:
            return res_wav
        inf_tau = np.where(np.abs(H) < tau)
        for i in inf_tau: res_wav[i] = H[i]*np.exp(nl * (np.abs(H[i]) - tau))
        return res_wav
    
    def run(self, level1, level2, wav, thrmeth=thresholding_fct, thrfct=XT):
        global nl, D, H, log_img 
        log_img = self.img_log
        tot = [level1, level2]
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
                leveln.append(thrfct(thrmeth()))
            new.append(leveln)

        leveln1 = new[0]
        leveln2 = new[1]
        An1 = pywt.idwt2((tot[1][0], (leveln2[0], leveln2[1], leveln2[2])), wav)[:, :tot[0][0].shape[1], :3]
        imglog = pywt.idwt2((An1, (leveln1[0], leveln1[1], leveln1[2])), wav)
        final_img = exposure.rescale_intensity(np.exp(imglog[: , :self.img.shape[1], :3]), out_range=(0,1))
        return final_img


if __name__=='__main__':
    img_path = "Dataset_BUSI_with_GT/benign/benign(341).png" #109, 423, 434, 421, 392, 341, 313
    img = plt.imread(img_path)
    final_img = Paper_approach(img).get_despeckled_image('db2', 'Paper_implementation')