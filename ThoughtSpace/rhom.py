from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from itertools import combinations
from itertools import permutations
from itertools import product

from ThoughtSpace.pca import basePCA

import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import copy
from factor_analyzer.rotator import Rotator

from scipy.linalg import orthogonal_procrustes

import matplotlib.pyplot as plt
from random import randint
import seaborn as sns
from scipy.stats import t, norm


def crazyshuffle(arr):
    '''
    Full-Mantel Shuffle
    -------------------
    Shuffles the rows and then columns of an inputted array.

    Parameters
    ----------
        arr: array-like or dataframe

    Returns
    -------
        array: shuffled array.
    '''
    arr = arr.loc[:, 'focus':].values
    x, y = arr.shape
    rows = np.indices((x,y))[0]
    cols = [np.random.permutation(y) for _ in range(x)]
    out = arr[rows, cols]
    return out

def bootstrap(estimator, X, y, group=None, cv=None, omnibus = False, splithalf = False, pro_cong = False, bypc = False, shuffle = False, fit_params={}):
    """
    Bootstrap Resampling
    --------------------
    This function performs bootstrap resampling on the provided data using the specified estimator and resampling method.

    Parameters
    ----------
        estimator : object
            The rhom object to use for fitting the data.
        X : array-like or DataFrame, shape (n_samples, n_features)
            The input data for the referent dataset.
        y : array-like, shape (n_samples,)
            The input data for the comparate dataset and the subject on which components are projected.
        group : array-like, shape (n_samples,), optional, default=None
            Grouping variable for shuffling.
        cv : object, default=None
            pair_cv object for method of resampling/cross-validating data.
        omnibus : bool, default=False.
            Whether to bootstrap omnibus-sample reproducibility.
        splithalf : bool, default=False.
            Whether to bootstrap split-half reproducibility.
        pro_cong : bool, default=False.
            Whether to include estimates of loading similarity with Tucker's Congruence Coefficient.
        bypc : bool, default=False.
            Whether to return omnibus-sample results by principal component.
        shuffle : bool, default=False.
            Whether to full-mantel shuffle the data.
        fit_params : dict, default=empty dictionary.
            Additional parameters to pass to the fit method of the estimator.

    Returns
    -------
        results : list
            A list containing the results of the bootstrap resampling.
    """
    # If shuffle is True, the data is shuffled using the specified group variable.
    if shuffle:
        firstcols_X = X.loc[:, group]

        Xdata = crazyshuffle(X)
        Xdata = pd.DataFrame(Xdata)
        X = pd.concat([firstcols_X, Xdata], axis = 1)

    avg_score = []
    exp_var = []

    # If pro_cong is True, procrustes congruence analysis is performed on the bootstrap samples.
    if pro_cong:
        avg_phi = []
        
    if not omnibus:
        # If bypc is True, omnibus-sample results are returned by principal component.
        if bypc:
            complist = []
            for i in range(estimator.n_comp):
                complist.append([])
            # If pro_cong is True, results include Tucker's Congruence Coefficients (TCC) in addition to R-Homologue scores.
            if pro_cong:
                philist = []
                for i in range(estimator.n_comp):
                    philist.append([])
            for x1, x2 in cv.bypc_split(X,y):
                
                ests = gen_ests(estimator, x1, x2, pro_cong = pro_cong)
                if pro_cong:
                    for i in range(len(philist)):
                        complist[i].append(ests[0][i])
                        philist[i].append(ests[1][i])
                else:
                    for i in range(len(complist)):
                        complist[i].append(ests[i])

        # If splithalf is True, split-half reliability is bootstrapped.
        elif splithalf:
        
            for x1, x2 in cv.redists(df=X, subset=y): 
                ests = gen_ests(estimator, x1, x2, pro_cong=pro_cong)
                if pro_cong:
                    avg_score.append(ests[0])
                    avg_phi.append(ests[1])
                else:
                    avg_score.append(ests)              

        # If neither split-half or omnibus-sample are selected, direct-projection reproducibility is bootstrapped.
        else:
            for x1, x2 in cv.split(X,y):

                ests = gen_ests(estimator, x1, x2, pro_cong = pro_cong)
                if pro_cong:
                    avg_score.append(ests[0])
                    avg_phi.append(ests[1])
                else:
                    avg_score.append(ests)
    
    # If omnibus is True, omnibus-sample reproducibility is bootstrapped.
    else:

        for x1, x2 in cv.redists(df=X, subset=y):

            ests = gen_ests(estimator, x1, x2, pro_cong=pro_cong)
            if pro_cong:
                avg_score.append(ests[0])
                avg_phi.append(ests[1])
            else:
                avg_score.append(ests)

    if bypc:
        if pro_cong:
            return [complist, philist]
        
        return complist
    
    if pro_cong:
        return [avg_score, avg_phi, exp_var]

    return avg_score

def gen_ests(estimator, x1, x2, pro_cong = False, fit_params={}):
    """
    Generate component-similarity estimates using the provided estimator.

    This function fits the estimator to the provided data subsets 'x1' and 'x2', generates predictions,
    calculates correlations, and computes homologous pairs. Optionally, it performs procrustes congruence
    analysis.

    Parameters
    ----------
        estimator : object
            The rhom object to use for fitting the data.
        x1 : array-like or DataFrame, shape (n_samples, n_features)
            The referent subset of input data.
        x2 : array-like or DataFrame, shape (n_samples, n_features)
            The comparate subset of input data.
        pro_cong : bool, default=False.
            Whether to generate estimates of loading similarity using the Tucker's Congruence Coefficient.
        fit_params : dict, default=empty dictionary.
            Additional parameters to pass to the fit method of the estimator.

    Returns
    -------
        results : list or float
            If pro_cong is True, a list containing the results of homologous pairs and procrustes congruence analysis. If pro_cong is False, the results of homologous pairs.
    """
    # The estimator is fitted to the provided data subsets using the rhom .fit method.
    estimator.fit(x1,x2,**fit_params)
    # Predictions are generated using the rhom .predict method.
    preds = estimator.predict()
    # Correlation coefficients are calculated between the predicted values.
    corrs = np.corrcoef(preds[0], preds[1], rowvar=False)
    # Homologous pairs are computed based on the correlation matrix of x1 and x2 using the rhom .hom_pairs method.
    rhm = estimator.hom_pairs(corrs)
    # If pro_cong is True, procrustes congruence analysis is performed using the rhom .pro_cong method.
    if pro_cong:
        phi = estimator.pro_cong()
        return [rhm, phi]

    return rhm

def tcc(fac1=None, fac2=None):
    """
    Tucker's Congruence Coefficient
    -------------------------------
    Calculate the Tucker's Congruence Coefficient (TCC) between two components, which is an estimate of their loading similarity.

    Parameters
    ----------
        fac1 : array-like
            The first component.
        fac2 : array-like
            The second component.

    Returns
    -------
        tcc : float
            The Tucker's Congruence Coefficient (TCC) between the two factors.

    Notes
    -----
        - The TCC is a measure of loading similarity between two given components (Tucker, 1951).
        - Lovik et al. (2020) suggest using the absolute value of the numerator for factor matching.

    References
    ----------
        Tucker, L. R. (1951). A method for synthesis of factor analysis studies (PRS-984). Washington, DC: Department of the Army. 

        Lovik, A., Nassiri, V., Verbeke, G., & Molenberghs, G. (2020). A modified tucker’s congruence coefficient for factor matching.
            Methodology: European Journal of Research Methods for the Behavioral and Social Sciences,
            16(1), 59-74. https://doi.org/10.5964/meth.2813 

    """
    numerator = np.sum(abs(fac1*fac2))
    denominator = np.sqrt(np.sum(fac1**2) * np.sum(fac2**2))
    return numerator / denominator

class rhom(BaseEstimator):
    """
    R-Homologue and Tucker's Congruence Coefficient
    -----------------------------------------------
    A class for calculating component similarity between two sets of data.

    Parameters
    ----------
        - rd: array-like or pd.DataFrame, default=None
            - The referent dataset for deriving components.
        - n_comp: int, default=None
            - The number of components to extract from the dataset.
        - method: str, default="svd"
            - Method for PCA implementation.
            - Can take:
            1. 'svd' (sklearn PCA(svd_solver = "full"), numpy, scipy, MATLAB, R (prcomp), Stata)
            2. 'eigen'(SPSS, R (psych, princomp), SAS, sklearn PCA(svd_solver="auto"))
        - rotation: str, default="Varimax"
            - The rotation method to use for the loadings. If None or False, no rotation is performed.
            - Supported methods are "varimax", "promax", "oblimin", "oblimax", "quartimin", "quartimax", and "equamax".
        - bypc: bool, default=False
            - Whether to save component similarity on a by-component basis.

    Attributes
    ----------
        - model_x: PCA
            - The PCA object for the referent dataset.
        - model_x2: PCA
            - The PCA object for the comparate dataset.
        - results: pd.DataFrame
            - The projected component scores for each set of components.

    Methods
    -------
        - get_params(deep=True):
            - Retrieve the parameters of the instance.

        - fit(x, y=None):
            - Fit the model to the provided referent and comparate data.

        - predict(y=None):
            - Predict the output based on the provided input data.

        - hom_pairs(cor_matrix):
            - Calculate the correlation of homologous pairs in the correlation matrix of two datasets \n\t\t(Mulholland et al., 2023; Everett, 1983).

        - pro_cong():
            - Perform procrustes congruence analysis.
   
    References
    ----------

    Mulholland, B., Goodall-Halliwell, I., Wallace, R., Chitiz, L., McKeown, B., Rastan, A., Poerio, G. L., \n\tLeech, R., Turnbull, A., Klein, A., Milham, M., Wammes, J. D., Jefferies, E., & Smallwood, J. (2023). \n\tPatterns of ongoing thought in the real world. Consciousness and Cognition, 114, 103530. https://doi.org/10.1016/j.concog.2023.103530 

    Everett, J. E. (1983). Factor Comparability As A Means Of Determining The Number Of Factors And Their Rotation. \n\tMultivariate Behavioral Research, 18(2), 197-218. https://doi.org/10.1207/s15327906mbr1802_5

    """
    def __init__(self, rd=None, n_comp=None, method='svd', rotation="varimax", bypc=False):
        self.rd = rd
        self.n_comp = n_comp
        self.rotation = rotation
        self.bypc = bypc
        self.method = method

    
    def get_params(self,deep=True):
        """
        Retrieve the parameters of the instance.

        Parameters
        ----------
            deep: bool, optional
                If True, returns a deep copy of the parameters. If False, returns a shallow copy. Defaults to True.

        Returns
        -------
            dict: 
                A dictionary containing the parameters of the instance. If deep is True, a deep copy of the parameters is returned, otherwise, a shallow copy is returned.
        """
        return (copy.deepcopy(
            {"rd":self.rd})
            if deep
            else {"rd":self.rd})
        
    def fit(self, x, y=None):
        """
        Fit the model to the provided data.

        This function fits the model to the provided input data 'x' and 'y' if applicable.

        Parameters
        ----------
            x : array-like or DataFrame, shape (n_samples, n_features)
                The input data for the referent dataset.
            y : array-like or DataFrame, shape (n_samples, n_targets), optional, default=None
                The input data for the comparate dataset.

        Returns
        -------
            None
        """

        # If k >= 2, a basePCA model is fitted to the 'referent' data with the specified rotation method.
        if self.n_comp >= 2:
                self.model_x = basePCA(n_components=self.n_comp, verbosity = 0, rotation=self.rotation, method=self.method)

        # If k < 2, a basePCA model is fitted to the 'referent' data without rotation.
        else:
            self.model_x = basePCA(n_components=self.n_comp, verbosity = 0, rotation=False, method=self.method)

        self.model_x.fit(pd.DataFrame(x))

        # Regardless of the value of k, a basePCA model is always fitted to the 'comparate' data without rotation.
        self.model_x2 = basePCA(n_components=self.n_comp, verbosity = 0, rotation=False, method = self.method)
        self.model_x2.fit(pd.DataFrame(y))

    def predict(self, y=None):
        """
        Generate component scores based on projections from each dataset.

        Parameters
        ----------
            y : array-like, default=None
                The input data for prediction.

        Returns
        -------
            results : list
                A list containing the predicted outputs.
        """
        y = self.rd

        #If rotation is specified, the prediction is performed by calculating the dot product of the comparate set with the loadings of the fitted PCA models ('model_x' and 'model_x2') after applying orthogonal Procrustes rotation.
        if self.rotation :
            results = []

            loadings = self.model_x.loadings.to_numpy()

            R, s = orthogonal_procrustes(loadings, self.model_x2.loadings.to_numpy())
            loadings_x2 = np.dot(self.model_x2.loadings.to_numpy(), R.T)

            for loads in [loadings, loadings_x2]:
                self.results = np.dot(y, loads)
                results.append(self.results)

        #If no rotation, the prediction is performed by transforming the comparate set y using the loadings of the fitted PCA models ('model_x' and 'model_x2').
        else:
            results = []
            for model in [self.model_x,self.model_x2]:
                preds = model.transform(self.rd)
                results.append(preds)
    
        return results

    def hom_pairs(self,cor_matrix):
        """
        Calculate homologous components between the referent and comparate datasets.

        Parameters
        ----------
            cor_matrix : array-like, shape (n_features, n_features)
                The correlation matrix between the components for the referent and comparate datasets.

        Returns
        -------
            homologous_pairs : list or float
                If self.bypc is True, a list containing the correlation values of homologous pairs. If self.bypc is False, the mean correlation value of homologous pairs.
        """

        #The function first extracts the relevant cells from the correlation matrix.
        cor_matrix = cor_matrix[-self.n_comp:,:self.n_comp]
        cor_matrix = np.abs(cor_matrix)

        x = np.argmax(cor_matrix, axis = 0)
        x = [[en,a] for en,a in enumerate(x)]
        x2 = np.argmax(cor_matrix, axis = 1)
        x2 = [[a,en] for en,a in enumerate(x2)]
        x2 = sorted(x2, key = lambda x: x[0])

        #If the rows and columns with the highest correlations are different, it explores all possible permutations of the indices to find the arrangement that maximizes the mean correlation value.
        if x != x2:
            idx = list(range(self.n_comp))
            sols = list(permutations(idx, r=self.n_comp))
            soldict = {}
            for perm in sols:
                plis = []
                outls = []
                for e, cell in enumerate(perm):
                    plis.append([e,cell])

                    outs = cor_matrix[e,cell]
                    outls.append(outs)
                out = np.mean(outls)
                soldict[out] = plis
            
            bestval = max(list(soldict.keys()))

            #If self.bypc is True, it returns the correlation values of homologous pairs.
            if self.bypc:
                bestorg = soldict[bestval]
                bestorg = sorted(bestorg, key = lambda x: x[0])    
                rhoms = [cor_matrix.T[z[0],z[1]] for z in bestorg]              
                return rhoms
            
            
            #If self.bypc is False, it returns the mean correlation value of homologous pairs.
            else:
                return bestval

        else:
            rhoms = [cor_matrix.T[z[0],z[1]] for z in x]
            if self.bypc:
                return rhoms
            else:
                return np.mean(rhoms)

    def pro_cong(self):
        """
        Performs procrustes congruence analysis between the loadings matrices of the referent and comparate fitted PCA models. It calculates the Tucker's Congruence Coefficient (TCC) for all possible combinations of loadings pairs between the two models.

        Returns
        -------
            phi : list or float
                If self.bypc is True, a list containing the TCC values of homologous pairs. If self.bypc is False, the mean TCC value of homologous pairs.

        """
        loadings_X = self.model_x.loadings.to_numpy()
        loadings_x2 = self.model_x2.loadings.to_numpy()
        
        #The function first applies orthogonal Procrustes rotation to align the loadings matrices.
        R, s = orthogonal_procrustes(loadings_X, loadings_x2)
        loadings_x2 = np.dot(loadings_x2, R.T)

        #It then calculates the TCC between each pair of loadings vectors from the aligned matrices.
        tcc_matrix = []
        tcclist = []
        for i in range(len(loadings_X.T)):
            for j in range(len(loadings_x2.T)):
                phi = tcc(loadings_X.T[i], loadings_x2.T[j])
                tcclist.append(phi)
            tcc_matrix.append(tcclist)
            tcclist =[]
        
        tcc_matrix = np.asarray(tcc_matrix)

        #The resulting TCC matrix is passed to the 'hom_pairs' method to determine the homologous pairs.
        phi = self.hom_pairs(tcc_matrix)
        return phi

class pair_cv():
    """
    Resampling Methods for Bootstrapping Component Reproducibility Analysis
    ------------------------------------------------------------------------
    A class for bootstrap resampling component-similarity analyses.

    Parameters
    ----------
        - k: int, default=5
            - Number of folds to cross-validate direct-projection reproducibility.
        - n: int, default=1000
            - Number of resamples to bootstrap similarity scores.
        - boot: bool, default=False
            - Whether to bootstrap direct-projection reproducibility by comparing every combination of folds in the referent dataset with every combination of folds in the comparate.
        - omnibus: bool, default=False
            - Whether to bootstrap resample according to omnibus-sample reproducibility or split-half reliability.
        - group: str, default=False
            - The name of the column to be the selection variable by which to conduct separate reprodcubility analyses.
        
    Attributes
    ----------
        - scaler: StandardScaler
            - The scaler used for z-score normalization.

    Methods
    -------
        - divide_chunks(l, c):
            - Divides inputted dataset into specified number of folds.

        - assignModel(df=None):
            - Assigns rows of inputted dataset to either an 'omnibus' or 'sample' subset. See omni_sample() for more detail.

        - standardize(df=None):
            - z-score normalizes an inputted dataframe using scaler.

        - omni_prep(df=None):
            - Used in by-component omnibus-sample reproducibility. Assigns 'omnibus' and 'sample' subsets for each level of the grouping variable. See bypc() for more detail.

        - omni_prep_mini(df=None, subsamps=None, subset=None):
            - Used in regular omnibus-sample reproducibility. Assigns 'omnibus' and 'sample' subsets for a specific level of the grouping variable. See omni_sample() for more detail.
        
        - split(X, y):
            - Used in direct-projection reproducibility. Splits the referent and comparate dataframes into folds. If 'boot' is set to True, it will compare every combination of folds in the referent dataset with every combination of folds in the comparate. 

        - bypc_split(X, y):
            - Used in by-component omnibus-sample reproducibility. Splits the established omnibus and sample sets into folds and compares every combination of folds in the omnibus dataset with every combination of folds in the sample set.

        - split_half(df=None):
            - Used in split-half reliability. Randomly assigns the rows of an inputted dataframe into one of two halves.

        - split_frame(df):
            - Used in tandem with split_half(). Divides the assigned halves into split dataframes.

        - redists(df=None, subset=None):
            - Bootstrap resamples the split-half or omnibus-sample subdivisions of an inputted dataframe, outputting a list of iterations.
        """
    def __init__(self, k=5, n=1000, boot=False, omnibus=False, group=None) :
        self.n_splits = k
        self.n_redists = n
        self.boot = boot
        self.omnibus = omnibus
        self.group = group

    def divide_chunks(self, l, c) :
        """
        Divide a list into chunks of approximately equal size.

        Parameters
        ----------
            l: list
                The list to be divided into chunks.
            c: int
                The desired number of chunks.

        Yields
        ------
            list:
                A generator that yields chunks of the original list, with each chunk containing approximately equal elements.
        """
        n = len(l)//c
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def assignModel(self, df=None, subrows=None) :
        """
        Randomly assign rows of a dataframe to be part of 'omnibus' or 'sample' subsets.

        Parameters
        ----------
            - df: pd.DataFrame, default=None
                - The DataFrame to which model types will be assigned. If not provided, an empty DataFrame is created.
            - subrows: int, default=None
                - To ensure each subset of the dataframe contributes an equal number of rows to the omnibus solution,
                this argument limits the number of rows assigned to the 'omnibus' set to always be equal to the size of half of
                the subset with the smallest number of rows.
                - So if subset A includes 1000 rows, and subset B includes 800 rows,
                both subset A and B will contribute 800/2 = 400 rows to the omnibus solution and their remaining rows to
                their respective sample solutions. 
            
        Returns
        -------
            pd.DataFrame:
                The DataFrame with the 'o/s' column indicating the assigned 'omnibus/sample' rows.

        """
        rows = df.shape[0]
        df['o/s'] = "omnibus"
        df['o/s'][0:int(subrows)] = "sample"
        df['o/s'] = np.random.permutation(df['o/s'].values)
        return df

    def standardize(self, df=None) :
        """
        Standardize the values in the DataFrame using z-score normalization.

        Parameters
        ----------
            df: pd.DataFrame, default=None
                The DataFrame whose values will be standardized.

        Returns
        -------
            pd.DataFrame:
                The DataFrame with standardized values.

        Notes
        -----
            - Standardization is performed independently on each feature (column) of the DataFrame.
            - This function uses the StandardScaler from scikit-learn for standardization.
        """
        scaler = StandardScaler()
        dft = scaler.fit_transform(df)
        df = pd.DataFrame(dft,index=df.index,columns=df.columns)
        return df
    
    def omni_prep(self, df=None, subrows=None) :
        """
        Prepares data for by-component omnibus-sample reproducibility by partitioning, assigning omnibus/sample grouping to rows, and standardizing.

        Parameters
        ----------
            - df: pd.DataFrame, default=None
                - The DataFrame containing the data to be prepared for omnibus-sample reproducibility analysis.
            - subrows: int, default=None
                - Limits the number of rows contributed to omnibus solution according to the subset with the smallest sample. See the assignModel() docstring for more detail.

        Returns
        -------
            dict:
                A dictionary containing each sample dataframe and the omnibus dataframe.
        """
        # The 'group' attribute specifies the column in the DataFrame used for grouping the data.
        samples = df[self.group].unique()
        subsamps = {}
        for sample in samples:
            subsamps[str(sample)] = df[df[self.group] == sample]

        models = {"omnibus":pd.DataFrame()}

        # This function depends on the 'assignModel' and 'standardize' methods of the class.
        for subsamp in subsamps:
            model = subsamps[subsamp]
            model = self.assignModel(model, subrows = subrows)

            # The 'o/s' column in each DataFrame represents the assigned model type ('sample' or 'omnibus').
            models[subsamp] = model[model["o/s"] == "sample"]
            models[subsamp] = models[subsamp].drop(labels = [self.group, "o/s"], axis = 1)
            models[subsamp] = self.standardize(models[subsamp])

            models['omnibus'] = models['omnibus'].append(model[model["o/s"] == "omnibus"])

        models['omnibus'] = models['omnibus'].drop(labels = [self.group, "o/s"], axis = 1)
        models['omnibus'] = self.standardize(models['omnibus'])
        
        return models

    def omni_prep_mini(self, df=None, subsamps=None, subset=None, subrows=None) :
        """
        Prepare data for omnibus-sample reproducibility analysis with a specified subset.

        Parameters
        ----------
            - df: pd.DataFrame, default=None
                - The DataFrame containing the data to be prepared for omnibus-sample analysis.
            - subsamps: dict, default=None
                - A dictionary containing subsets of the data named by their corresponding levels of the grouping variable.
            - subset: str, default=None
                - The name of the subset to be processed.
            - subrows: int, default=None
                - Limits the number of rows contributed to omnibus solution according to the subset with the smallest sample. See the assignModel() docstring for more detail.

        Returns
        -------
            dict:
                A dictionary containing the omnibus and sample subdivisions of the dataframe.
        """
        # The 'group' attribute specifies the column in the DataFrame used for grouping the data.
        model = df[df[self.group] == subset]
        model = self.assignModel(model, subrows=subrows)

        # The 'o/s' column in each DataFrame represents the assigned model type ('sample' or 'omnibus').
        models = {"omnibus":model[model["o/s"] == "omnibus"]}

        models[subset] = model[model["o/s"] == "sample"]
        models[subset] = models[subset].drop(labels = [self.group, "o/s"], axis = 1)
        models[subset] = self.standardize(models[subset])

        # This function depends on the 'assignModel' and 'standardize' methods of the class.
        for subsamp in subsamps:
            model = subsamps[subsamp]
            model = self.assignModel(model, subrows=subrows)

            models['omnibus'] = models['omnibus'].append(model[model["o/s"] == "omnibus"])

        models['omnibus'] = models['omnibus'].drop(labels = [self.group, "o/s"], axis = 1)
        models['omnibus'] = self.standardize(models['omnibus'])

        return models

    def split(self, X, y) :
        """
        Generate indices to split data into referent and comparate sets for cross-validation and/or bootstrapping.

        Parameters
        ----------
            X: array-like or DataFrame
                The referent data.
            y: array-like or Series
                The comparate data.

        Yields
        ------
            tuple:
                A tuple containing the indices of the referent and comparate sets for each fold.
        """
        foldidx = list(range(self.n_splits))
        try:
            X = X.values
            y = y.values

        except:
            pass

        # This function shuffles the referent features and comparate values.
        np.random.shuffle(X)
        np.random.shuffle(y)
        x1_c = list(self.divide_chunks(X, self.n_splits))
        x2_c = list(self.divide_chunks(y, self.n_splits))

        folds = []
        # If bootstrapping is not enabled, it pairs the fold indices for referent and comparate sets.
        if not self.boot:
            for fold in product(foldidx, repeat=2):
                folds.append(fold)
            folds = [[x1_c[z],x2_c[q]] for z,q in folds]

        # If bootstrapping is enabled, it generates all combinations of fold indices for referent and comparate sets.
        elif self.boot:
            for z in range(1,self.n_splits+1):
                for fold in combinations(foldidx, r=z):
                    folds.append(fold)
                
            boot_folds = []
            for fold in product(folds, repeat=2):
                boot_folds.append(fold)

            Xs1 = [list(x[0]) for x in boot_folds]
            Xs2 = [list(x[1]) for x in boot_folds]

            fold_chks = []
            folds1 = []
            for z in Xs1:
                fold_chks.append([x1_c[v] for v in z])
                folds1.append(np.concatenate((fold_chks[0]), axis=0).squeeze())
                fold_chks =[]

            fold_chks = []
            folds2 = []
            for z in Xs2:
                fold_chks.append([x2_c[v] for v in z])
                folds2.append(np.concatenate((fold_chks[0]), axis=0).squeeze())
                fold_chks =[]    
            folds = [list(z) for z in zip(folds1,folds2)]
        
        for x1,x2 in folds:
            yield x1,x2

    def bypc_split(self, X, y):
        """
        Generate indices to split data into referent and comparate sets based on unique values of a grouping variable.

        Parameters
        ----------
            X: array-like or DataFrame
                The referent features.
            y: array-like or Series
                The comparate values.

        Yields
        ------
            tuple:
                A tuple containing the indices of the referent and comparate sets for each fold.
        """
        foldidx = list(range(self.n_splits))
        try:
            X = X.values
            y = y.values

        except:
            pass

        # This function shuffles the comparate values.
        np.random.shuffle(y)
        x_c = list(self.divide_chunks(y, self.n_splits))
        folds = []
        # If bootstrapping is not enabled, it pairs the fold indices for referent and comparate sets.
        if not self.boot:
            for fold in product(foldidx, repeat=2):
                folds.append(fold)
                folds = [[X,x_c[z]] for z in folds]

        # If bootstrapping is enabled, it generates all combinations of fold indices for referent and comparate sets.
        elif self.boot:
            for z in range(1,self.n_splits+1):
                for fold in combinations(foldidx, r=z):
                    folds.append(fold)
                
            boot_folds = []
            for fold in product(folds, repeat=2):
                boot_folds.append(fold)

            Xs = [list(x[1]) for x in boot_folds]

            fold_chks = []
            folds = []
            for z in Xs:
                fold_chks.append([x_c[v] for v in z])
                folds.append([X, np.concatenate((fold_chks[0]), axis=0).squeeze()])
                fold_chks =[]
        
        for x1,x2 in folds:
            yield x1,x2
  
    def split_half(self, df=None) :
        """
        Split the DataFrame randomly into two halves.

        Parameters
        ----------
            df: pd.DataFrame, default=None
                The DataFrame to be split.

        Returns
        -------
            pd.DataFrame:
                The DataFrame with a new column 'subset' indicating the assigned halves.
        """
        rows = df.shape[0]
        df['subset'] = 2
        df['subset'][0:int(rows/2)] = 1

        return df
        
    def split_frame(self, df) :
        """
        Split the DataFrame into two separate DataFrames based on the 'subset' column.

        Parameters
        ----------
            df: pd.DataFrame, default=None
                The DataFrame to be split.

        Returns
        -------
            dict:
                A dictionary containing two DataFrames, with keys '1' and '2', representing the subsets.
        """
        models = {'1':pd.DataFrame(), '2':pd.DataFrame()}

        for model in models:
            models[model] = df[df["subset"] == int(model)]
            models[model] = models[model].drop(labels = ["subset"], axis = 1)
            models[model] = self.standardize(models[model])

        df.drop(labels = ["subset"], axis = 1)
        
        return models
    
    def redists(self, df=None, subset=None) :
        """
        Generates a set of bootstrap reassignments of different subdivisions of either omnibus/sample or split-half reproducibility.

        Parameters
        ----------
            df: pd.DataFrame, default=None
                The DataFrame containing the data to be redistributed.
            subset: str, default=None 
                The subset of the grouping variable for which to generate a set of redistributions.

        Returns
        -------
            list:
                A list of redistributions, where each redistribution is represented as a list of two DataFrames.
        """
        redists = []
        if self.omnibus:
            samples = df[self.group].unique()
            subsamps = {}
            for sample in samples:
                if sample != subset :
                    subsamps[sample] = df[df[self.group] == sample]

            nrows = df[self.group].value_counts().reset_index()
            nrows.columns = [self.group, 'frequency']
            nval = (nrows['frequency'].min())/2

            for i in range(self.n_redists):
                redist = self.omni_prep_mini(df=df, subset=subset, subsamps=subsamps, subrows=nval)
                redistv = list(redist.values())
                redists.append(redistv)
        
        else:
            if self.group != None:
                splitdf = df[df[self.group] == subset].copy()
                splitdf = splitdf.drop(labels=self.group, axis=1)
                splitdf = self.split_half(splitdf.copy())
            else:
                splitdf = self.split_half(df.copy())

            for i in range(self.n_redists):
                splitdf['subset'] = np.random.permutation(splitdf['subset'].values)
                redist = self.split_frame(splitdf)
                redistv = list(redist.values())
                redists.append(redistv)  

        return redists    

def check_paths(parent_folder, prefix):
    path = os.getcwd()
    fullpath = os.path.join(path, parent_folder)
    if not os.path.exists(fullpath):
        os.makedirs(fullpath, exist_ok = True)

    project_path = os.path.join(fullpath, prefix)
    if not os.path.exists(project_path):
        os.makedirs(project_path, exist_ok = True)

def splithalf(df=None, group=None, npc=None,
              method = 'svd', rotation='varimax', boot=1000, 
              save=True, display=False, shuffle=False, 
              path = 'results', file_prefix=randint(10000,99999)) :
    '''
    Split-Half Reliability
    ----------------------
    This function conducts a bootstrapped split-half reliability analysis
    on your dataframe. It can do so on a full dataset, or at each level of a
    grouping variable. It simply bootstrap reassigns random halves of the data
    into two subsets and computes their component similarity based on:
        1) Loading similarity (with Tucker's Congruence Coefficient: Tucker, 1951; See also Lovik et al., 2020)
        2) Component-score similarity (with R-homologue: Mulholland et al., 2023; See also Everett, 1983)

    Parameters
    ----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        npc: int, default=None
            Number of components to extract per solution.
        
        rotation: str, default="varimax"
            Rotation method to be performed on referent. "none" for no rotation.

        boot: int, default=1000
            Number of bootstrap samples to generate 95% confidence intervals.
        
        save: bool, default=True
            Save outputted split-half reliability to .csv.

        display: bool, default=False
            Print output in the terminal.

        shuffle: bool, default=False
            Perform analysis on shuffled "garbage" data.

        path: str, default='results'
            The path to the output directory.

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files. By default will classify files with random 5-digit ID.

    Returns
    -------
        pd.DataFrame:
            The function at minimum returns a pandas dataframe with the results.
        
        .csv:
            If save=True, will save /results to a csv.

        printed results:
            If display=True, prints the output directly in the terminal.
    '''
    if group != None :
        df_t = df.drop(labels = [group], axis = 1)
        samples = df[group].unique()

    else :
        df_t = df
        samples = ['fulldata']

 
    boot_model = rhom(rd = copy.deepcopy(df_t.values),
                      n_comp = npc,
                      method = method,
                      rotation=rotation)
        
    cv = pair_cv(group=group, n=boot)

    if save:
        check_paths(path, file_prefix)

    split_df = pd.DataFrame(columns=['n_comp', group, 'rhm_x','rhm_sd','rhm_LCI','rhm_UCI','phi_x','phi_sd','phi_LCI','phi_UCI'])

    def getstats(data):
        conf_int = norm.interval(0.95, np.median(data), scale = np.std(data))
        conf_int = list(conf_int)
        if conf_int[1] > 1:
            conf_int[1] = 1
        mean = np.mean(data)
        sd = np.std(data)
        return conf_int, mean, sd
    
    for i in range(len(samples)):
        
        print('Running ' + str(samples[i]))
        resultslist = bootstrap(boot_model, df, samples[i], cv=cv, splithalf = True, pro_cong=True, shuffle=shuffle)

        print('Saving ' + str(samples[i]))
        #scaler = StandardScaler()
        #s_results = scaler.fit_transform(np.array(results).reshape(-1, 1)).squeeze()
        rhm_ci, rhm_x, rhm_sd = getstats(resultslist[0])
        phi_ci, phi_x, phi_sd = getstats(resultslist[1])
 
        stats_dict = {}
        stats_dict['n_comp'] = str(boot_model.n_comp) + "PC"
        stats_dict[group] = str(samples[i])

        stats_dict['rhm_x'] = rhm_x
        stats_dict['rhm_sd'] = rhm_sd
        stats_dict['rhm_LCI'] = rhm_ci[0]
        stats_dict['rhm_UCI'] = rhm_ci[1]

        stats_dict['phi_x'] = phi_x
        stats_dict['phi_sd'] = phi_sd
        stats_dict['phi_LCI'] = phi_ci[0]
        stats_dict['phi_UCI'] = phi_ci[1]

        newrow = pd.DataFrame(stats_dict, index = [0])
        split_df = pd.concat([split_df, newrow], axis = 0)

        if display:

            print("Split-Half Reliability for " + str(samples[i]) + ": ")
            print("*"*20)
            for results in resultslist:
                if results == resultslist[0]:
                    print(f"Mean Homologue Similarity: {rhm_x:.3g} +/- {rhm_sd:.3g} 95% CI[{rhm_ci[0]:.3g}, {rhm_ci[1]:.3g}]")
                else:
                    print(f"Mean Factor Congruence: {rhm_x:.3g} +/- {rhm_sd:.3g} 95% CI[{rhm_ci[0]:.3g}, {rhm_ci[1]:.3g}]")

            print("*"*40)

    if save:

        split_df.to_csv(os.path.join(path, f"{file_prefix}/{file_prefix}_splithalf_{(len(df_t.columns))}D_{(boot_model.n_comp)}PC.csv"), index = False)
        
        print('dataframe saved')

    return split_df  

def dir_proj(df=None, group=None, npc=None,
             method='svd', rotation="varimax", folds=5,
             save=True, plot=True, display=False, shuffle=False, 
             path = 'results', file_prefix=randint(10000,99999)):    
    '''
    Direct-Projection Reproducibility
    ---------------------------------
    This function conducts a bootstrapped direct-projection analysis
    on your data based on some inputted grouping variable. This involves
    dividing each group into its own dataframe, and assessing the similarity
    of the components generated by each group to each other group based on:
        1) Loading similarity (with Tucker's Congruence Coefficient: Tucker, 1951; See also Lovik et al., 2020)
        2) Component-score similarity (with R-homologue: Mulholland et al., 2023; See also Everett, 1983)
    
    Parameters
    ----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        npc: int, default=None
            Number of components to extract per solution.
        
        rotation: str, default="varimax"
            Rotation method to be performed on referent. "none" for no rotation.

        folds: int, default=5
            Number of folds to use for cross-validation.
        
        save: bool, default=True
            Save outputted reproducibility results to .csv.

        plot: bool, default=True
            Visualise results with heatmaps.

        display: bool, default=False
            Print output in the terminal.

        shuffle: bool, default=False
            Perform analysis on shuffled "garbage" data.

        path: str, default='results'
            The path to the output directory.

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files. By default will classify files with random 5-digit ID.

    Returns
    -------
        pd.DataFrame:
            The function at minimum returns a pandas dataframe with the results.
    
        .csv:
            If save=True, will save a .csv file to /results.

        .png:
            If plot=True, will save heatmaps for loading similarity and component-score similarity.

        printed results:
            If display=True, prints the output directly in the terminal.
    '''
    groups = df[group].unique()

    #create a data frame dictionary to store your data frames
    maindict = {elem : pd.DataFrame() for elem in groups}

    for key in maindict.keys():
        maindict[key] = df[:][df[group] == key]
        maindict[key] = maindict[key].drop(labels=group, axis=1)

    dalist = groups
    dagoodlist = list(combinations(dalist, 2))

    scaler = StandardScaler()
    df = df.drop(labels=group, axis=1)
    dft = scaler.fit_transform(df)
    df = pd.DataFrame(dft,index=df.index,columns=df.columns)

    if save:
        check_paths(path, file_prefix)

    dirproj_df = pd.DataFrame(columns=['n_comp', 'referent', 'comparator', 'rhm_x','rhm_sd','rhm_LCI','rhm_UCI','phi_x','phi_sd','phi_LCI','phi_UCI'])

    if plot:
        dirproj_mtx = pd.DataFrame(columns=dalist, index=dalist)
        dirproj_phi = pd.DataFrame(columns=dalist, index=dalist)

    def getstats(data):
        conf_int = norm.interval(0.95, np.median(data), scale = np.std(data))
        conf_int = list(conf_int)
        if conf_int[1] > 1:
            conf_int[1] = 1
        mean = np.mean(data)
        sd = np.std(data)
        return conf_int, mean, sd

    boot_model = rhom(rd = copy.deepcopy(df.values),
                      n_comp = npc, 
                      method = method,
                      rotation=rotation)
    
    cv = pair_cv(boot=True, k=folds)
    for currentset in dagoodlist:
        print("Running " + str(currentset[0]) + " x " + str(currentset[1]))
        resultslist = bootstrap(boot_model, maindict[currentset[0]],maindict[currentset[1]], cv=cv, pro_cong = True, shuffle=shuffle)
        #scaler = StandardScaler()
        #s_results = scaler.fit_transform(np.array(results).reshape(-1, 1)).squeeze()
        print("Saving " + str(currentset[0]) + " x " + str(currentset[1]))
        rhm_ci, rhm_x, rhm_sd = getstats(resultslist[0])
        phi_ci, phi_x, phi_sd = getstats(resultslist[1])
    
            
        stats_dict = {}

        stats_dict['n_comp'] = str(boot_model.n_comp) + "PC"
        stats_dict['referent'] = str(currentset[0])
        stats_dict['comparator'] = str(currentset[1])

        stats_dict['rhm_x'] = rhm_x
        stats_dict['rhm_sd'] = rhm_sd
        stats_dict['rhm_LCI'] = rhm_ci[0]
        stats_dict['rhm_UCI'] = rhm_ci[1]

        stats_dict['phi_x'] = phi_x
        stats_dict['phi_sd'] = phi_sd
        stats_dict['phi_LCI'] = phi_ci[0]
        stats_dict['phi_UCI'] = phi_ci[1]

        newrow = pd.DataFrame(stats_dict, index = [0])
        dirproj_df = pd.concat([dirproj_df, newrow], axis = 0)

        if plot:

            dirproj_mtx.loc[str(currentset[0]), str(currentset[1])] = rhm_x
            dirproj_mtx.loc[str(currentset[1]), str(currentset[0])] = rhm_x

            dirproj_phi.loc[str(currentset[0]), str(currentset[1])] = phi_x
            dirproj_phi.loc[str(currentset[1]), str(currentset[0])] = phi_x
        
        if display:

            print(f"Mean Homologue Similarity: {rhm_x:.3g} +/- {rhm_sd:.3g} 95% CI[{rhm_ci[0]:.3g}, {rhm_ci[1]:.3g}]")
            print(f"Mean Factor Congruence: {phi_x:.3g} +/- {phi_sd:.3g} 95% CI[{phi_ci[0]:.3g}, {phi_ci[1]:.3g}]")

            print("*"*40)

    if plot:
        dirproj_mtx = dirproj_mtx.fillna(1)
        dirproj_phi = dirproj_phi.fillna(1)

        plt.close()

        sns.heatmap(dirproj_mtx,
                    vmin = dirproj_mtx.values.min(),
                    annot = True,
                    annot_kws={"fontsize": 35 / np.sqrt(len(dirproj_mtx))},
                    cmap = "flare")
        plt.suptitle('Mean Homologue Similarity', fontsize=16)
        plt.savefig(os.path.join(path, f"{file_prefix}/{file_prefix}_heatmap{(len(df.columns))}D_{boot_model.n_comp}PC_rhm.png"))
        plt.show()
        plt.close()


        sns.heatmap(dirproj_phi,
                    vmin = dirproj_phi.values.min(),
                    annot = True,
                    annot_kws={"size": 35 / np.sqrt(len(dirproj_phi))},
                    cmap = "flare")
        plt.suptitle('Mean Factor Congruence', fontsize=16)
        plt.savefig(os.path.join(path, f"{file_prefix}/{file_prefix}_heatmap{(len(df.columns))}D_{boot_model.n_comp}PC_phi.png"))
        plt.show()
        plt.close()

    if save:
        dirproj_df.to_csv(os.path.join(path, f"{file_prefix}/{file_prefix}_dj{len(df.columns)}D_{boot_model.n_comp}PC.csv"), index = False)

    return dirproj_df

def omni_sample(df=None, group=None, npc=None,
                method='svd', rotation="varimax", boot=1000,
                save=True, display=False, shuffle=False,
                path = 'results', file_prefix=randint(10000,99999)):
    '''
    Omnibus-Sample Reproducibility
    ------------------------------
    This function conducts an omnibus-sample reproducibility analysis on your data.
    It randomly bootstrap reassigns halves of each level of an inputted grouping variable 
    to be used in either a 'sample' or 'omnibus' subset. The 'sample' subsets generate
    components representative of that level of the grouping variable, while the 'omnibus'
    subsets are aggregated with other groups to produce 'common' components. The analysis
    assesses the component similarity of the orthogonal aggregated set relative to each sample.
    It computes component similarity with:
        1) Loading similarity (with Tucker's Congruence Coefficient: Tucker, 1951; See also Lovik et al., 2020)
        2) Component-score similarity (with R-homologue: Mulholland et al., 2023; See also Everett, 1983)

    Parameters
    ----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        npc: int, default=None
            Number of components to extract per solution.
        
        rotation: str, default="varimax"
            Rotation method to be performed on omnibus set. "none" for no rotation.

        boot: int, default=1000
            Number of bootstrap samples to generate 95% confidence intervals.
        
        save: bool, default=True
            Save outputted omnibus-sample reliability to .csv.

        display: bool, default=False
            Print output in the terminal.

        shuffle: bool, default=False
            Perform analysis on shuffled "garbage" data.

        path: str, default='results'
            The path to the output directory.

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files. By default will classify files with random 5-digit ID.

    Returns
    -------
        pd.DataFrame:
            The function at minimum returns a pandas dataframe with the results.
        
        .csv:
            If save=True, will save /results to a csv.

        printed results:
            If display=True, prints the output directly in the terminal.
    '''
    samples = df[group].unique()

    df_t = df.drop(labels = [group], axis = 1)

    omsamp_df = pd.DataFrame(columns=['n_comp', group, 'rhm_x','rhm_sd','rhm_LCI','rhm_UCI','phi_x','phi_sd','phi_LCI','phi_UCI'])

    if save:
        check_paths(path, file_prefix)

    def getstats(data):
        conf_int = norm.interval(0.95, np.median(data), scale = np.std(data))
        conf_int = list(conf_int)
        if conf_int[1] > 1:
            conf_int[1] = 1
        mean = np.mean(data)
        sd = np.std(data)
        return conf_int, mean, sd

 
    boot_model = rhom(rd = copy.deepcopy(df_t.values),
                      n_comp = npc,
                      method = method,
                      rotation=rotation)
 
        
    cv = pair_cv(omnibus = True, group=group, n=boot)
    totalRhm = []
    totalPhi = []
    for i in range(len(samples)):
        print("Running omnibus x " + str(samples[i]))
        resultslist = bootstrap(boot_model, df, samples[i], cv=cv, omnibus=True, pro_cong=True, shuffle = shuffle)

        print("Saving omnibus x " + str(samples[i]))
        totalRhm.append(resultslist[0])
        totalPhi.append(resultslist[1])

        rhm_ci, rhm_x, rhm_sd = getstats(resultslist[0])
        phi_ci, phi_x, phi_sd = getstats(resultslist[1])

        stats_dict = {}

        stats_dict['n_comp'] = int(boot_model.n_comp)
        stats_dict[group] = str(samples[i])

        stats_dict['rhm_x'] = rhm_x
        stats_dict['rhm_sd'] = rhm_sd
        stats_dict['rhm_LCI'] = rhm_ci[0]
        stats_dict['rhm_UCI'] = rhm_ci[1]

        stats_dict['phi_x'] = phi_x
        stats_dict['phi_sd'] = phi_sd
        stats_dict['phi_LCI'] = phi_ci[0]
        stats_dict['phi_UCI'] = phi_ci[1]

        newrow = pd.DataFrame(stats_dict, index = [0])
        omsamp_df = pd.concat([omsamp_df, newrow], axis = 0)

        if display:
            print('Omnibus x ' + str(samples[i]) + ':')
            print("*"*20)
            print(f"Mean Homologue Similarity: {rhm_x:.3g} +/- {rhm_sd:.3g} 95% CI[{rhm_ci[0]:.3g}, {rhm_ci[1]:.3g}]")
            print(f"Mean Factor Congruence: {phi_x:.3g} +/- {phi_sd:.3g} 95% CI[{phi_ci[0]:.3g}, {phi_ci[1]:.3g}]")
            print("*"*40)

    print('Saving overall results')
    rhm_ci, rhm_x, rhm_sd = getstats(totalRhm)
    phi_ci, phi_x, phi_sd = getstats(totalPhi)

    if display:
        print("*"*20)
        print(f"Mean Homologue Similarity: {rhm_x:.3g} +/- {rhm_sd:.3g} 95% CI[{rhm_ci[0]:.3g}, {rhm_ci[1]:.3g}]")
        print(f"Mean Factor Congruence: {phi_x:.3g} +/- {phi_sd:.3g} 95% CI[{phi_ci[0]:.3g}, {phi_ci[1]:.3g}]")
        print("*"*40)

    stats_dict = {}

    stats_dict['n_comp'] = int(boot_model.n_comp)
    stats_dict[group] = "Total"

    stats_dict['rhm_x'] = rhm_x
    stats_dict['rhm_sd'] = rhm_sd
    stats_dict['rhm_LCI'] = rhm_ci[0]
    stats_dict['rhm_UCI'] = rhm_ci[1]

    stats_dict['phi_x'] = phi_x
    stats_dict['phi_sd'] = phi_sd
    stats_dict['phi_LCI'] = phi_ci[0]
    stats_dict['phi_UCI'] = phi_ci[1]

    newrow = pd.DataFrame(stats_dict, index = [0])
    omsamp_df = pd.concat([omsamp_df, newrow], axis = 0)

    if save:

        omsamp_df.to_csv(os.path.join(path, f"{file_prefix}/{file_prefix}_omsamp_{len(df_t.columns)}D_{boot_model.n_comp}PC.csv"), index = False)

    return omsamp_df

def bypc(df=None, group=None, npc=None,
         method='svd', rotation="varimax", folds=5,
         save=True, plot=True, display=False, shuffle = False,
         path = 'results', file_prefix=randint(10000,99999)):
    '''
    Omnibus-Sample Reproducibility: By-Component
    ------------------------------
    This function conducts an omnibus-sample reproducibility analysis on your data,
    modified to assess the correspondence between each component of an omnibus solution and its
    corresponding components in each subset. It randomly reassigns halves of each level 
    of an inputted grouping variable to be stably used in either a 'sample' or 'omnibus' subset. 
    The 'sample' subsets are folded to generate cross-validated components representative of that level of
    the grouping variable, while the 'omnibus' subset is aggregated with other groups to produce 'common' components.
    The analysis assesses the component similarity of the orthogonal aggregated set relative to each sample.
    It computes component similarity with:
        1) Loading similarity (with Tucker's Congruence Coefficient: Tucker, 1951; See also Lovik et al., 2020)
        2) Component-score similarity (with R-homologue: Mulholland et al., 2023; See also Everett, 1983)

    Parameters
    ----------

        df: pd.Dataframe, default=None
            It should include only the columns to be decomposed and your grouping variable.

        group: str, default=None
            The column heading for your grouping variable.

        npc: int, default=None
            Number of components to extract per solution.
        
        rotation: str, default="varimax"
            Rotation method to be performed on omnibus set. "none" for no rotation.

        folds: int, default=5
            Number of folds to use for cross-validation.
        
        save: bool, default=True
            Save outputted omnibus-sample reliability to .csv.

        plot: bool, default=True
            Save wordclouds, a Scree plot, and .csv files for the specified omnibus set.

        display: bool, default=False
            Print output in the terminal.

        shuffle: bool, default=False
            Perform analysis on shuffled "garbage" data.

        path: str, default='results'
            The path to the output directory. 

        file_prefix: str, default=randint(10000,99999)
            Provide name to distinguish saved files. By default will classify files with random 5-digit ID.

    Returns
    -------
        pd.DataFrame:
            The function at minimum returns a pandas dataframe with the results.
        
        .csv:
            If save=True, will save /results to a csv.

        thoughtspace PCA results:
            If plot=True, will save the results, including wordclouds and Scree plot, for the omnibus set in a thoughtspace folder.

        printed results:
            If display=True, prints the output directly in the terminal.
    '''
    df_t = df.drop(labels=group, axis=1)
    
    boot_model = rhom(rd = copy.deepcopy(df_t.values),
                      bypc=True, n_comp=npc,
                      method = method,
                      rotation = rotation)
    
    cv = pair_cv(boot = True, group=group, k=folds)

    nrows = df[group].value_counts().reset_index()
    nrows.columns = [group, 'frequency']
    nval = (nrows['frequency'].min())/2

    maindict = cv.omni_prep(df = df, subrows = nval)

    samples = df[group].unique()
    dagoodlist=[]
    for i in range(len(samples)):
        sampset = ['omnibus', samples[i]]
        dagoodlist.append(sampset)

    def getstats(data):
        conf_int = norm.interval(0.95, np.median(data), scale = np.std(data))
        conf_int = list(conf_int)
        if conf_int[1] > 1:
            conf_int[1] = 1
        mean = np.mean(data)
        sd = np.std(data)
        return conf_int, mean, sd
    
    if save:
        check_paths(path, file_prefix)

    stats_bypc = pd.DataFrame(columns=['n_comp', group, 'comp', 'rhm_x','rhm_sd','rhm_LCI','rhm_UCI','phi_x','phi_sd','phi_LCI','phi_UCI'])

    for currentset in dagoodlist:
        comps = bootstrap(boot_model, maindict[currentset[0]], maindict[currentset[1]], cv=cv, bypc = True, pro_cong = True, shuffle=shuffle)

        print("Running " + str(currentset[0]) + " x " + str(currentset[1]))
        # print("*" * 20)
        for i in range(len(comps[0])):
            stats_dict = {}
            rhm_ci, rhm_x, rhm_sd = getstats(comps[0][i])
            phi_ci, phi_x, phi_sd = getstats(comps[1][i])

            stats_dict['n_comp'] = str(boot_model.n_comp) + "PC"
            stats_dict[group] = str(currentset[1])
            stats_dict['comp'] = i + 1

            stats_dict['rhm_x'] = rhm_x
            stats_dict['rhm_sd'] = rhm_sd
            stats_dict['rhm_LCI'] = rhm_ci[0]
            stats_dict['rhm_UCI'] = rhm_ci[1]

            stats_dict['phi_x'] = phi_x
            stats_dict['phi_sd'] = phi_sd
            stats_dict['phi_LCI'] = phi_ci[0]
            stats_dict['phi_UCI'] = phi_ci[1]

            newrow = pd.DataFrame(stats_dict, index = [0])
            stats_bypc = pd.concat([stats_bypc, newrow], axis = 0)

            if display:
                print(f"Component {i + 1}:")
                print(f"Homologue Similarity: {rhm_x:.3g} +/- {rhm_sd:.3g} 95% CI[{rhm_ci[0]:.3g},{rhm_ci[1]:.3g}]")
                print(f"Factor Congruence: {phi_x:.3g} +/- {phi_sd:.3g} 95% CI[{phi_ci[0]:.3g},{phi_ci[1]:.3g}]")
                print("*"*40)

    if plot:
        model = basePCA(n_components=npc, rotation=rotation)
        # model.fit(maindict['omnibus'])
        model.fit_transform(maindict['omnibus'])
        model.save(path=os.path.join(path, f"{file_prefix}"), pathprefix=f"{file_prefix}_os")

    if save:
        stats_bypc.to_csv(os.path.join(path, f"{file_prefix}/{file_prefix}_bypc_{len(df_t.columns)}D_{boot_model.n_comp}PC.csv"), index = False)

    return stats_bypc