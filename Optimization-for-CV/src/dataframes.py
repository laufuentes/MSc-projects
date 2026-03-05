import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

class df_9: 
    def __init__(self, df, wav): 
        self.df = df 
        self.wav = wav
        self.new = self.new_df()
        self.new_cols = self.new.columns
        self.max, self.max_cols = self.three_bests()
        pass

    def new_df(self): 
        #Definition of a new dataframe gathering results from the average of each threshold function and type combination
        new = pd.DataFrame()
        thr = self.df['Threshold_type'].unique()
        thr_fct = self.df['Threshold_function'].unique()

        k = 0
        new_cols = []
        for i,t in enumerate(thr): 
            for j,m in enumerate(thr_fct): 
                new_cols.append(t+'_'+m)
                k += 1 
                new[k]= self.df[(self.df['Threshold_type']==t) & (self.df['Threshold_function']==m)].iloc[:, 3:].min()

        new.columns = new_cols
        self.new = new
        new.to_csv("images/"+self.wav+"/results_"+self.wav+".csv")
        return new

    def three_bests(self): 
        max = np.unique(np.argsort(self.new.to_numpy(), axis=1)[(0,1,3), :3]).tolist()
        mse = np.argsort(self.new.to_numpy(), axis=1, kind='mergesort')[2, :3]
        [max.append(mse.tolist()[i]) for i in range(3)]
        max = np.unique(max)
        max_cols = [self.new_cols[i] for i in max]
        return max, max_cols 

    def plt_Threshold_comparison_several(self): 
        fig, ax = plt.subplots(nrows = self.new.shape[0], figsize=(30,2*self.new.shape[0]))
        plt.suptitle('Metric values per method')

        for i in range(self.new.shape[0]):
            yy = (self.new.iloc[i]).to_numpy()
            sns.barplot(x=self.new.columns[0:], y=yy[0:], ax=ax[i], hue=self.new.columns[0:])
            ax[i].set_ylabel(yy[0])
            ax[i].set_xlabel((self.new.iloc[i]).name)

            
        plt.tight_layout()
        plt.savefig('images/Report/Threshold_comparison_several.png')
        plt.show()
        pass

    def plt_Threshold_comparison(self): 
        #Among each thresholding function type we select the best performances 
        best = self.new[self.max_cols]

        sns.scatterplot(data=best)
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.savefig('images/Report/Threshold_comparison.png')
        plt.show()
        pass 


class df_Filters: 
    def __init__(self, df, methods_name): 
        self.df = df 
        self.new = self.summary_df(methods_name)
        self.methods_name = methods_name
        pass

    def summary_df(self, methods_name): 
        new = pd.DataFrame()
        for i,w in enumerate(methods_name): 
            new[i] = self.df[self.df['method']==w].iloc[:, 2:].mean()

        new.columns = methods_name 
        new.to_csv('images/Report/filters_summary.csv')
        return new
    
    def plt_Filters(self): 
        sns.lineplot(data=self.new)
        plt.savefig('images/Report/Filters.png')
        plt.show()
        pass 

    def plt_Different_filters(self): 
        fig, ax = plt.subplots(ncols = self.new.shape[0], figsize=(10,3*self.new.shape[0]))
        plt.suptitle('Metric values per method')
        colors = ['r', 'b', 'g', 'y']

        for i in range(self.new.shape[0]):
            yy = (self.new.iloc[i]).to_numpy()
            sns.barplot(x=self.new.columns[0:], y=yy[0:], label=yy[0], ax=ax[i], hue=self.new.columns[0:])
            ax[i].set_xlabel((self.new.iloc[i]).name)
            plt.setp(ax[i].xaxis.get_majorticklabels(), rotation=45, ha='right')  # Rotate x-axis labels


        plt.tight_layout()
        plt.savefig('images/Report/Different_filters.png')
        plt.show()
        pass

class df_wavelets: 
    def __init__(self, df) -> None:
        self.df = df
        self.wavs = ['db1', 'db2', 'sym2', 'bior1.1']
        self.new = self.summary()
        pass

    def summary(self): 
        new = pd.DataFrame()
        for i,w in enumerate(self.wavs): 
            new[i] = self.df[self.df['wavelet']==w].iloc[:, 2:].mean()

        new.columns = self.wavs 
        new.to_csv('images/wavelets/results.csv')
        return new
    
    def plt_wavelets_performance(self):
        sns.scatterplot(data=self.new).set
        plt.savefig('images/Report/wavelets_performance.png')
        plt.show() 
        pass

    def plt_Metric_wavelet_performance(self): 
        fig, ax = plt.subplots(ncols = self.new.shape[0], figsize=(50,2*self.new.shape[0]))
        plt.suptitle('Metric values per method')

        for i in range(self.new.shape[0]):
            yy = (self.new.iloc[i]).to_numpy()
            sns.barplot(x=self.new.columns[0:], y=yy[0:], ax=ax[i], hue=self.new.columns[0:])
            ax[i].set_xlabel((self.new.iloc[i]).name)
            plt.setp(ax[i].xaxis.get_majorticklabels(), rotation=45, ha='right')  # Rotate x-axis labels


        plt.tight_layout()
        plt.savefig('images/Report/Metric_wavelet_performance.png')
        plt.show()
        pass
