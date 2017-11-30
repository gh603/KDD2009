from sklearn.decomposition import PCA 
import numpy as np
def draw_pca(trainX_dummy):
    x=trainX_dummy.shape[1]
    pca=PCA()
    pca.fit_transform(trainX_dummy)
    exp_var=pca.explained_variance_ratio_
    
    cum_exp_var = np.cumsum(exp_var)
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(10,5))
        plt.plot(range(1,101), cum_exp_var[:100], 'r--')
        plt.axis([1, 100, 0, 1])
        plt.xticks(np.arange(1, 100, 5.0))
        plt.yticks(np.arange(0, 1, 0.05))
        plt.ylabel('Accumulated explained variance ratio')
        plt.xlabel('Principal components')
        plt.show()
        
    
#pca_visualize(trainX_dummy)