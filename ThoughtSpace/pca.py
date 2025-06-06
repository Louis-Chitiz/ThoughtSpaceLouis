from typing import Tuple
import numpy as np
import pandas as pd

from factor_analyzer import Rotator, calculate_bartlett_sphericity, calculate_kmo
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from scipy.stats import pearsonr
from scipy.linalg import eigh

from sklearn.preprocessing import StandardScaler

from ThoughtSpace.plotting import save_wordclouds, plot_scree,plot_stats
from ThoughtSpace.utils import setupanalysis, returnhighest, clean_substrings
import os

from sklearn.model_selection import KFold, BaseCrossValidator

class basePCA(TransformerMixin, BaseEstimator):
    """
    A base class for performing Principal Component Analysis (PCA) with ThoughtSpace.

    Args:
        n_components (int or "infer", optional): The number of components to keep. If "infer", the number of components is determined based on the explained variance. Defaults to "infer".
        verbosity (int, optional): The level of verbosity. Set to 0 for no output, 1 for basic output, and 2 for detailed output. Defaults to 1.
        rotation (str or bool, optional): The rotation method to use for the loadings. If False, no rotation is performed. Supported methods are "varimax", "promax", "oblimin", "oblimax", "quartimin", "quartimax", and "equamax". Defaults to "varimax".

    Attributes:
        n_components (int or "infer"): The number of components to keep.
        verbosity (int): The level of verbosity.
        rotation (str or bool): The rotation method to use for the loadings.
        path (str or None): The path to save the results.
        ogdf (pd.DataFrame or None): The original dataframe.
        scaler (StandardScaler): The scaler used for z-score normalization.
        loadings (pd.DataFrame): The loadings matrix.
        extra_columns (pd.DataFrame): The fitted PCA scores.
        project_columns (pd.DataFrame): The projected PCA scores.
        _raw_fitted (pd.DataFrame): The raw fitted data.
        _raw_project (pd.DataFrame): The raw projected data.
        fullpca (PCA): The PCA object with full components.
        items (list): The column names of the input dataframe.

    Methods:
        check_stats(df: pd.DataFrame) -> None:
            This function checks the KMO and Bartlett Sphericity of the dataframe.

        check_inputs(df: pd.DataFrame, fit: bool = False, project: bool = False) -> pd.DataFrame:
            Check the inputs of the function.

        z_score(df: pd.DataFrame) -> np.ndarray:
            This function returns the z-score of the dataframe.

        naive_pca(df: pd.DataFrame) -> Tuple[PCA, pd.DataFrame]:
            Perform PCA on the input dataframe.

        fit(df, y=None, scale: bool = True, **kwargs) -> "PCA":
            Fits the PCA model.

        transform(df: pd.DataFrame, scale=True) -> pd.DataFrame:
            Transform the input dataframe using the fitted PCA model.

        save(group=None, path=None, pathprefix="analysis", includetime=True) -> None:
            Save the results of the PCA analysis.

    """
    def __init__(self, n_components="infer",verbosity=1,rotation="varimax", method='svd'):
        self.n_components = n_components
        self.verbosity = verbosity
        self.rotation = rotation
        self.method = method
        self.path = None
        self.ogdf = None


        
    def check_stats(self, df: pd.DataFrame) -> None:
        """
        This function checks the KMO and Bartlett Sphericity of the dataframe.
        Args:
            df: The dataframe to check.
        Returns:
            None
        """
        if self.verbosity > 0:
            bart = calculate_bartlett_sphericity(df)
            n, p = df.shape
            print (f"The matrix shape is ({n}, {p}).")
            degreesfreedom = p * (p - 1) / 2
            if bart[1] < 0.05:
                print(f"Bartlett Sphericity is acceptable. χ2({int(degreesfreedom)}) = %.2f." % bart[0],"The p-value is %.3f" % bart[1])
            else:
                print(
                    "Bartlett Sphericity is unacceptable. Something is very wrong with your data. The p-value is %.3f"
                    % bart[1]
                )
            kmo = calculate_kmo(df)
            k = kmo[1]
            if k < 0.5:
                print(
                    "KMO score is unacceptable. The value is %.2f, you should not trust your data."
                    % k
                )
            if 0.6 > k > 0.5:
                print(
                    "KMO score is miserable. The value is %.2f, you should consider resampling or continuing data collection."
                    % k
                )
            if 0.7 > k > 0.6:
                print(
                    "KMO score is mediocre. The value is %.2f, you should consider continuing data collection, or use the data as is."
                    % k
                )
            if 0.8 > k > 0.7:
                print(
                    "KMO score is middling. The value is %.2f, your data is perfectly acceptable, but could benefit from more sampling."
                    % k
                )
            if 0.9 > k > 0.8:
                print(
                    "KMO score is meritous. The value is %.2f, your data is perfectly acceptable."
                    % k
                )
            if k > 0.9:
                print(
                    "KMO score is marvelous. The value is %.2f, what demon have you sold your soul to to collect this data? Please email me."
                    % k
                )

    def check_inputs(
        self, df: pd.DataFrame, fit: bool = False, project: bool = False
    ) -> pd.DataFrame:
        """
        Check the inputs of the function.
        Args:
            df: The input dataframe.
            fit: Whether the function is in fit mode.
        Returns:
            The processed dataframe.
        """
        if fit:
            self.extra_columns = df.copy()
        if project:
            self.project_columns = df.copy()
        if isinstance(df, pd.DataFrame):
            dtypes = df.dtypes
            for col in dtypes.index:
                if fit and dtypes[col] in [np.int64, np.float64, np.int32, np.float32]:
                    self.extra_columns.drop(col, axis=1, inplace=True)
                if project and dtypes[col] in [np.int64, np.float64, np.int32, np.float32]:
                    self.project_columns.drop(col, axis=1, inplace=True)
                if dtypes[col] not in [np.int64, np.float64, np.int32, np.float32]:
                    df.drop(col, axis=1, inplace=True)
            if fit:
                self.items = df.columns.tolist()
        else:
            self.items = [f"item_{x}" for x in range(df.shape[1])]
            self.extra_columns = pd.DataFrame()
        return df

    def z_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        This function returns the z-score of the dataframe.
        Args:
            df: The dataframe to be scaled.
        Returns:
            The z-score of the dataframe.
        """
        self.scaler = StandardScaler()
        return pd.DataFrame(self.scaler.fit_transform(df))

    # def naive_pca(self, df: pd.DataFrame) -> Tuple[PCA, pd.DataFrame]:  # type: ignore
    #     """
    #     Perform Principal Component Analysis (PCA) on the input dataframe.

    #     Args:
    #         df (pd.DataFrame): The dataframe to be used for PCA.

    #     Returns:
    #         Tuple[PCA, pd.DataFrame]: A tuple containing the PCA object and the loadings dataframe.

    #     Raises:
    #         TypeError: If the rotation type is not supported.

    #     """
    #     if self.n_components == "infer":
    #         self.fullpca = PCA(svd_solver="full").fit(df)
    #         self.n_components = len([x for x in self.fullpca.explained_variance_ if x >= 1])
    #         if self.verbosity > 0:
    #             print(f"Inferred number of components: {self.n_components}")
    #     else:
    #         self.fullpca = PCA(svd_solver="full").fit(df)
    #     pca = PCA(n_components=self.n_components,svd_solver="full").fit(df)
        
    #     if self.rotation == False:
    #         loadings = pca.components_.T
    #     elif self.rotation in ["varimax","promax","oblimin","oblimax","quartimin","quartimax","equamax"]:
    #         loadings = Rotator(method=self.rotation).fit_transform(pca.components_.T)
    #     else:
    #         raise "Rotation type is not supported"
        
    #     loadings = pd.DataFrame(
    #         loadings,
    #         index=self.items,
    #         columns=[f"PC{x+1}" for x in range(self.n_components)],
    #     )
    #     averages = loadings.mean(axis=0).to_dict()
    #     for col in averages:
    #         if averages[col] < 0:
    #             if self.verbosity > 1:
    #                 print(f"Component {col} has mostly negative loadings, flipping component")
    #             loadings[col] = loadings[col] * -1
    #     return loadings

    def naive_pca(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Principal Component Analysis (PCA) on the input NumPy array.

        Args:
            data (np.ndarray): The input data matrix (samples × variables).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - The rotated (or unrotated) component loadings (variables × components).
                - The explained variance of each component.
        """


        if self.method == "svd":
            # Fit PCA using sklearn (default approach)
            self.fullpca = PCA(svd_solver="full").fit(df)
            explained_variance = self.fullpca.explained_variance_

            if self.n_components == "infer":
                self.n_components = np.sum(explained_variance >= 1)
                if self.verbosity > 0:
                    print(f"Inferred number of components: {self.n_components}")

            pca = PCA(n_components=self.n_components, svd_solver="full").fit(df)
            loadings = pca.components_.T  # Unrotated component loadings
            explained_variance = pca.explained_variance_

        elif self.method == "eigen":

            # Step 2: Compute the correlation matrix (R)
            R = np.corrcoef(df, rowvar=False)

            # Step 3: Perform eigen decomposition of the correlation matrix
            self.fullpca, eigenvectors = eigh(R)  # ascending order
            idx = np.argsort(self.fullpca)[::-1]  # Descending order

            # Step 4: Select top n_components
            self.eigenvalues, eigenvectors = self.fullpca[idx], eigenvectors[:, idx]
            selected_eigenvectors = eigenvectors[:, :self.n_components]

            # Step 5: Compute unrotated component loadings
            loadings = selected_eigenvectors * np.sqrt(self.eigenvalues[:self.n_components])

            # # Step 6: Apply Varimax rotation with Kaiser normalization
            # rotator = Rotator(method=self.rotation, normalize=True)
            # rotated_loadings = rotator.fit_transform(loadings)

        else:
            raise ValueError("Invalid method. Choose 'svd' or 'eigen'.")

        if self.rotation == False:
            loadings = loadings
        elif self.rotation in ["varimax","promax","oblimin","oblimax","quartimin","quartimax","equamax"]:
            loadings = Rotator(method=self.rotation, normalize=True).fit_transform(loadings)
        else:
            raise "Rotation type is not supported"

        for i in range(loadings.shape[1]):
            if np.sum(loadings[:, i]) < 0:
                loadings[:, i] *= -1 

        loadings = pd.DataFrame(
            loadings,
            index=self.items,
            columns=[f"PC{x+1}" for x in range(self.n_components)],
        )

        # # Flip loadings if necessary
        # averages = loadings.mean(axis=0).to_dict()
        # for col in averages:
        #     if averages[col] < 0:
        #         if self.verbosity > 1:
        #             print(f"Component {col} has mostly negative loadings, flipping component")
        #         loadings[col] = loadings[col] * -1

        return loadings


    def fit(self, df, y=None, scale: bool = True, **kwargs) -> "PCA":
        """
        Fit the PCA model.
        Args:
            df: The input dataframe.
            y: The target variable.
            **kwargs: The keyword arguments.
        Returns:
            The fitted PCA model.
        """
        _df = df.copy()
        if isinstance(df,pd.DataFrame):
            dfidx = _df.index
        else:
            dfidx = list(range(len(_df)))
        try:
            cols = _df.columns
            outcols = []
            for col in cols:
                if "focus" in col.lower():
                    col = col.lower().replace("focus","Task")
                if "other" in col.lower():
                    col = col.lower().replace("other","People")
                outcols.append(col)
            mapper = {cols[x]:outcols[x] for x in range(len(outcols))}
            _df = _df.rename(mapper,axis=1)

        except:
            pass
        if self.ogdf is None:
            self.ogdf = _df.copy()
        _df = self.check_inputs(_df, fit=True)
        self.check_stats(_df)
        self._raw_fitted = _df
        if scale:
            _df = self.z_score(_df)
        self.loadings = self.naive_pca(_df)
       
        indivloadings = np.dot(_df,self.loadings).T
        for x in range(self.n_components):
            self.extra_columns[f"PCA_{x + 1}"] = indivloadings[x, :]
        self.extra_columns.index = dfidx

        if self.verbosity > 0:
            self.print_explained_variance()
            self.print_highestloadings()

        return self
    
    def print_explained_variance(self) -> None:
        """
        Print explained variance for each principal component or cumulative variance.
        """
        if self.verbosity > 0:

            if self.method == 'svd':
                if self.fullpca is not None:
                    explained_variance = self.fullpca.explained_variance_ratio_
                    cumulative_variance = self.fullpca.explained_variance_ratio_.cumsum()

                    print(f"The first {self.n_components} components explained {cumulative_variance[self.n_components-1]*100:.2f}% of cumulative variance.")
                    
                    if self.rotation:
                        pass  
                    else:
                        for i in range(self.n_components):
                            print(f"Component {i+1} explained {explained_variance[i]*100:.2f}% of variance.")

            elif self.method == 'eigen':
                # Compute % variance explained
                # Compute % variance explained
                explained_variance = self.eigenvalues[:self.n_components]
                explained_variance = (explained_variance / np.sum(self.eigenvalues)) * 100
                cumulative_variance = np.cumsum(explained_variance)

                print(f"The first {self.n_components} components explained {cumulative_variance[self.n_components-1]:.2f}% of cumulative variance.")



            else:
                print("PCA model not fitted. Please fit the model first.")

    def print_highestloadings(self) -> None:
        if self.verbosity > 0:
            if self.loadings is not None:
                question_names = self.loadings.index.tolist()
                question_names[0] = question_names[0].split("_")[0]
                max_subst = clean_substrings(question_names)
                if max_subst is not None:
                    question_names = [x.replace(max_subst, "") for x in question_names]
                    self.loadings.index = question_names
                for col in self.loadings:
                    highest = returnhighest(self.loadings[col], 3)
                    print (f"{col}: {highest}")
            else:
                print("PCA model not fitted. Please fit the model first.")

    def transform(
        self,
        df: pd.DataFrame,
        scale=True,
    ) -> pd.DataFrame:
        """
        Transform the input dataframe using the fitted PCA model.

        Args:
            df (pd.DataFrame): The input dataframe to be transformed.
            scale (bool, optional): Whether to scale the input dataframe. Defaults to True.

        Returns:
            pd.DataFrame: The transformed dataframe.

        """
        _df = df.copy()
        if isinstance(_df, pd.DataFrame):
            newdfidx = _df.index
        else:
            newdfidx = list(range(len(_df)))
        try:
            cols = _df.columns
            outcols = []
            for col in cols:
                if "focus" in col.lower():
                    col = col.lower().replace("focus","Task")
                if "other" in col.lower():
                    col = col.lower().replace("other","People")
                outcols.append(col)
            mapper = {cols[x]:outcols[x] for x in range(len(outcols))}
            _df = _df.rename(mapper,axis=1)
        except:
            pass
        _df = self.check_inputs(_df,project=True)
        self._raw_project = _df
        if scale:
            _df = self.scaler.transform(_df)
        
        output_ = np.dot(_df, self.loadings).T
        if isinstance(self.project_columns, pd.DataFrame):
            for x in range(self.n_components):
                self.project_columns[f"PCA_{x}"] = output_[x, :]
        else:
            self.project_columns = output_.T
        if isinstance(self.project_columns, pd.DataFrame) : 
            self.project_columns.index = newdfidx
        return self.project_columns.copy()

    
    def cv(self,data,cv=None):
        if not cv:
            cv = KFold()
        else:
            assert isinstance(cv,BaseCrossValidator)
        baseline = self.fit_transform(data)
        folds = []
        correlations = []
        for x,y in cv.split(data):
            self.fit(data.iloc[x])
            out = self.transform(data.iloc[y])
            folds.append(out)
            outv = self.check_inputs(out).values.ravel()
            bv = self.check_inputs(baseline.iloc[y]).values.ravel()
            correlations.append(pearsonr(outv,bv)[0])
        return correlations, folds
            
    
    def save(self,group=None,path=None,pathprefix="analysis",font = "helvetica", includetime=True) -> None:
        """
        Save the results of the PCA analysis.

        Args:
            group (str or None, optional): The group name for saving results. If None, save results for all groups. Defaults to None.
            path (str or None, optional): The path to save the results. If None, use the default path. Defaults to None.
            pathprefix (str, optional): The prefix for the path. Defaults to "analysis".
            includetime (bool, optional): Whether to include the timestamp in the path. Defaults to True.

        Returns:
            None

        """
        if self.path is None:

            self.path = setupanalysis(path,pathprefix,includetime)

        if group is None:

            os.makedirs(os.path.join(self.path,"wordclouds"),exist_ok=True)
            os.makedirs(os.path.join(self.path,"csvdata"),exist_ok=True)
            os.makedirs(os.path.join(self.path,"screeplots"),exist_ok=True)
            os.makedirs(os.path.join(self.path, "descriptives"),exist_ok=True)
            
            save_wordclouds(self.loadings,os.path.join(self.path,"wordclouds"),font)
            self.project_columns.to_csv(os.path.join(self.path,"csvdata","projected_pca_scores.csv"))
            self.extra_columns.to_csv(os.path.join(self.path, "csvdata","fitted_pca_scores.csv"))
            
            
            if not self.extra_columns.index.equals(self.project_columns.index):
                newidx = self.project_columns.index.difference(self.extra_columns.index)
                self.full_columns = pd.concat([self.extra_columns,self.project_columns.loc[newidx]])
            else:
                self.full_columns = self.project_columns
                
            if not self._raw_fitted.index.equals(self._raw_project.index):
                newidx = self._raw_project.index.difference(self._raw_fitted.index)
                self._raw_full = pd.concat([self._raw_fitted,self._raw_project.loc[newidx]])
            else:
                self._raw_full = self._raw_project
                  
                
            self.full_columns.to_csv(os.path.join(self.path, "csvdata","full_pca_scores.csv"))
            self.loadings.to_csv(os.path.join(self.path,"csvdata","pca_loadings.csv"))
            pd.concat([self.ogdf, self.check_inputs(self.extra_columns)], axis=1).to_csv(os.path.join(self.path,"csvdata","pca_scores_original_format.csv"))
            plot_stats(self._raw_fitted,os.path.join(self.path, "descriptives", "fitted"))
            plot_stats(self._raw_project,os.path.join(self.path, "descriptives", "projected"))
            plot_stats(self._raw_full,os.path.join(self.path, "descriptives", "full"))
            plot_scree(self,os.path.join(self.path, "screeplots", "scree"))
        
        else:
            os.makedirs(os.path.join(self.path,f"wordclouds_{group}"),exist_ok=True)
            os.makedirs(os.path.join(self.path,f"csvdata_{group}"),exist_ok=True)
            os.makedirs(os.path.join(self.path,"screeplots"),exist_ok=True)

            save_wordclouds(self.loadings,os.path.join(self.path,f"wordclouds_{group}"))
            self.project_columns.to_csv(os.path.join(self.path,f"csvdata_{group}","projected_pca_scores.csv"))
            self.extra_columns.to_csv(os.path.join(self.path, f"csvdata_{group}","fitted_pca_scores.csv"))

            if not self.extra_columns.index.equals(self.project_columns.index):
                newidx = self.project_columns.index.difference(self.extra_columns.index)
                self.full_columns = pd.concat([self.extra_columns,self.project_columns.loc[newidx]])
            else:
                self.full_columns = self.project_columns

            if not self._raw_fitted.index.equals(self._raw_project.index):
                newidx = self._raw_project.index.difference(self._raw_fitted.index)
                self._raw_full = pd.concat([self._raw_fitted,self._raw_project.loc[newidx]])
            else:
                self._raw_full = self._raw_project

            self.full_columns.to_csv(os.path.join(self.path, f"csvdata_{group}","full_pca_scores.csv"))
            self.loadings.to_csv(os.path.join(self.path,f"csvdata_{group}","pca_loadings.csv"))
            pd.concat([self.ogdf, self.check_inputs(self.extra_columns)], axis=1).to_csv(os.path.join(self.path,f"csvdata_{group}","pca_scores_original_format.csv"))
            plot_stats(self._raw_fitted,os.path.join(self.path, "descriptives", f"fitted_{group}"))
            plot_stats(self._raw_project,os.path.join(self.path, "descriptives", f"projected_{group}"))
            plot_stats(self._raw_full,os.path.join(self.path, "descriptives", f"full_{group}"))
            plot_scree(self.fullpca,os.path.join(self.path, "screeplots", f"scree_{group}"))
        
        print(f"Saving done. Results have been saved to {self.path}")


class groupedPCA(basePCA):
    """
    A class for performing grouped Principal Component Analysis (PCA).

    Args:
        grouping_col: The column to group by.
        n_components: The number of components to use.
        **kwargs: Additional keyword arguments.

    Attributes:
        grouping_col: The column to group by.

    Methods:
        z_score_byitem(df_dict) -> pd.DataFrame:
            Calculate the z-score of the dataframe.

        z_score_byitem_project(df_dict) -> pd.DataFrame:
            Calculate the z-score of the dataframe for projection.

        fit(df, y=None, **kwargs) -> self:
            Fit the grouped PCA model.

        transform(df) -> pd.DataFrame:
            Transform the input dataframe using the fitted grouped PCA model.

        save(savebygroup=False, path=None, pathprefix="analysis", includetime=True) -> None:
            Save the results of the grouped PCA analysis.

    """
    def __init__(self, grouping_col=None, n_components="infer", **kwargs):
        super().__init__(n_components)
        self.grouping_col = grouping_col
        if grouping_col is None:
            raise ValueError("Must specify a grouping column.")

    def z_score_byitem(self, df_dict) -> pd.DataFrame:
        """
        This function is used to calculate the z-score of the dataframe.

        Args:
            df_dict (dict): Dictionary of dataframes.

        Returns:
            pd.DataFrame: Dataframe with z-score.
        """
        self.scalerdict = {}
        outdict = []
        for key, value in df_dict.items():
            scaler = StandardScaler()
            value_ = self.check_inputs(value, fit=True)
            value_scaled = scaler.fit_transform(value_)
            extcol = self.extra_columns.copy().assign(
                **dict(zip(self.items, value_scaled.T))
            )
            self.scalerdict[key] = scaler
            outdict.append(extcol)
            
        return pd.concat(outdict, axis=0)

    def z_score_byitem_project(self, df_dict) -> pd.DataFrame:
        """
        This function takes a dictionary of dataframes and returns a dataframe with z-scored values.

        Args:
            df_dict (dict): A dictionary of dataframes.

        Returns:
            pd.DataFrame: A dataframe with z-scored values.
        """
        outdict = []
        for key, value in df_dict.items():
            value_ = self.check_inputs(value, project=True)
            try:
                scaler = self.scalerdict[key]
            except Exception:
                print(
                    f"Encountered a group in the data that wasn't seen while fitting: {key}. New group will be zscored individually."
                )
                scaler = StandardScaler()
                scaler.fit(value_)
            value_scaled = scaler.transform(value_)
            extcol = self.project_columns.copy().assign(
                **dict(zip(self.items, value_scaled.T))
            )
            outdict.append(extcol)
        return pd.concat(outdict, axis=0)

    def fit(self, df: pd.DataFrame, y=None, **kwargs):
        """
        Fit the grouped PCA model.

        Args:
            df (pd.DataFrame): The dataframe to fit.
            y (pd.Series): The target variable.
            **kwargs: Additional keyword arguments.

        Returns:
            self

        """
        self.ogdf = df.copy()
        d = dict(tuple(df.groupby(self.grouping_col)))
        zdf = self.z_score_byitem(d)
        super().fit(zdf, y=y, scale=False, **kwargs)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input dataframe using the fitted grouped PCA model.

        Args:
            df (pd.DataFrame): The input dataframe to be transformed.

        Returns:
            pd.DataFrame: The transformed dataframe.

        """
        d = dict(tuple(df.groupby(self.grouping_col)))
        zdf = self.z_score_byitem_project(d)
        return super().transform(zdf, scale=False)
    
    def save(self,savebygroup=False,path=None,pathprefix="analysis",includetime=True):
        """
    Save the results of the grouped PCA analysis.

    Args:
        savebygroup (bool, optional): Whether to save results by group. Defaults to False.
        path (str or None, optional): The path to save the results. If None, use the default path. Defaults to None.
        pathprefix (str, optional): The prefix for the path. Defaults to "analysis".
        includetime (bool, optional): Whether to include the timestamp in the path. Defaults to True.

    Returns:
        None

    Raises:
        NotImplementedError: If saving by group is not yet implemented.

    """
        self.path = setupanalysis(path,"grouped_"+pathprefix,includetime)
        if savebygroup:
            raise NotImplementedError("Saving by group is not yet implemented.")
            
        else:
            super().save()