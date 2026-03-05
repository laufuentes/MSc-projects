# Unsupervised learning 

Laura Fuentes Vicente, Angel Reyero Lobo

In this master's course project centered around "unsupervised learning," our primary goals encompass the replication of key findings from the research paper titled "Learning from missing data with the binary latent block model." Additionally, we aim to provide a comprehensive computational overview of the Latent Block Model (LBM) tailored for Missing Not at Random (MNAR) scenarios.


## Installation

### Creation of a conda environment
To facilitate the implementation of the algorithms, we have developed guided notebooks, each requiring a kernel with pre-installed packages. To streamline this process, we recommend establishing a dedicated conda environment designed for this specific project. Two approaches to achieve this include:

#### 1- Run the following command in the terminal 
```
conda create --name NSA_FVRL --file requirements.txt
```
#### 2- Step-by-step: 
```
conda create --name NSA_FVRL
conda activate NSA_FVRL
conda install jupyter pandas numpy matplotlib scikit-learn tqdm seaborn argparse
conda install pytorch torchvision -c pytorch
```

### Kernel selection in python notebook
Before runing the notebook, we will need to select the predefined kernel.

## Usage

We have developed five notebooks to explore the code version of the Latent Block Model (LBM) for Missing Not at Random (MNAR) scenarios:

- **1.1-Dummy_training.ipynb:** This notebook initiates the training process for the Variational Expectation-Maximization (VEM) model, as proposed in the referenced article. It provides a preliminary exploration with a focus on one iteration of the VEM algorithm, offering insights into the model's early learning dynamics.

- **1.2-Model_LBM_MNAR.ipynb:** Provides an overview of the computation of the criterion $J(\gamma, \theta)$

- **2-Train.ipynb:** Designed to train the entire model on the parliament dataset. Given the potential time-intensive nature of this procedure, we have saved the parameters in the file named "trained_parameters.yaml", so that, computing this step is not required to continue exploring the notebooks. 

- **3-Figure_creation.ipynb:** Specifically crafted for creating figures 12, 17, and 18 from the article (saved in \Figures). Running this notebook does not necessitate the execution of the entire model, as parameters are loaded from the yaml file.

- **4-ICL.ipynb**: Designed to compute the ICL criterion associated to the trained model from 2-Train.ipynb

Given the potential computational expense of training, we recommend utilizing a GPU. To specify the device, the device argument can be employed, with 'cuda' recommended for general (use or 'mps' for Mac). 


The default configuration sets the number of row classes to 3 and column classes to 5. 

*NB: some of the parameters have been renamed:*
- C => P 
- D => Q
- $\gamma => (\nu_a, \nu_b, \nu_p, \nu_q, \rho_a, \rho_b,\rho_p,\rho_q, \tau_1, \tau_2)$
    - $tau^{Y} => tau_1$
    - $tau^{Z} => tau_2$
- $\theta => (\mu , \sigma_a^2, \sigma_b^2, \sigma_p^2, \sigma_q^2, \alpha_1, \alpha_2, \pi)$
    - $\alpha => \alpha_1$
    - $\beta => \alpha_2$

We based our work on the Github from the article: 
https://github.com/gfrisch/LBM-MNAR