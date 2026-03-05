import os
import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def new_df(df, wav): 
    #Definition of a new dataframe gathering results from the average of each threshold function and type combination
    new = pd.DataFrame()
    thr = df['Threshold_type'].unique()
    thr_fct = df['Threshold_function'].unique()

    k = 0
    new_cols = []
    for i,t in enumerate(thr): 
        for j,m in enumerate(thr_fct): 
            new_cols.append(t+'_'+m)
            k += 1 
            new[k]= df[(df['Threshold_type']==t) & (df['Threshold_function']==m)].iloc[:, 3:].min()

    new.columns = new_cols
    new.to_csv("images/synthetic/results_"+wav+".csv")
    return new

def three_bests(new, wav): 
        max = np.unique(np.argsort(new.to_numpy(), axis=1)[(0,1,3), :2]).tolist()
        mse = np.argsort(new.to_numpy(), axis=1, kind='mergesort')[2, :2]
        [max.append(mse.tolist()[i]) for i in range(2)]
        max = np.unique(max)
        max_cols = [new.columns[i] for i in max]
        best = new[max_cols]

        sns.scatterplot(data=best)
        plt.title(wav)
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.savefig('images/Report/synthetic/Threshold_comparison'+wav+'.png')
        plt.show()
        return best

def get_all_paths(folder_path):
    # Initialize an empty list to store paths
    all_paths = []
    # Iterate through all elements in the folder
    for root, _, files in os.walk(folder_path):
        # Add paths of files
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.split('.')[-1]=='png':
                all_paths.append(file_path)

    return all_paths

def edge_keeping_index(original_image, filtered_image):
    # Compute Laplacian of the original and filtered images
    laplacian_original = (cv2.Laplacian((original_image*255).astype(np.uint8), cv2.CV_64F)[:, :, :1]).reshape(original_image.shape[0:2])
    laplacian_filtered = (cv2.Laplacian((filtered_image*255).astype(np.uint8), cv2.CV_64F)[:, :, :1]).reshape(filtered_image.shape[0:2])


    # Compute the edge-keeping index
    num = np.sum(np.dot((laplacian_original - np.mean(laplacian_original)).T, (laplacian_filtered - np.mean(laplacian_filtered))))
    denom = np.sum(np.dot((laplacian_original - np.mean(laplacian_original)).T, (laplacian_original - np.mean(laplacian_original))))*np.sum(np.dot((laplacian_filtered - np.mean(laplacian_filtered)).T, (laplacian_filtered - np.mean(laplacian_filtered))))
    eki = num /np.sqrt(denom)

    return eki