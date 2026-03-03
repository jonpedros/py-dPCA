""" demixed Principal Component Analysis
"""

# Author: Wieland Brendel <wieland.brendel@neuro.fchampalimaud.org>
#
# License: BSD 3 clause

import numpy as np
from collections import OrderedDict
from itertools import combinations, chain
from scipy.linalg import pinv

from sklearn.base import BaseEstimator
import numexpr as ne
from utils import shuffle2D, denoise_mask, get_noise_covariance, nearest_centroid_accuracy

class dPCA(BaseEstimator):
    """ demixed Principal component analysis (dPCA)

    dPCA is a linear dimensionality reduction technique that automatically discovers
    and highlights the essential features of complex population activities. The
    population activity is decomposed into a few demixed components that capture most
    of the variance in the data and that highlight the dynamic tuning of the population
    to various task parameters, such as stimuli, decisions, rewards, etc.

    Parameters
    ----------
    labels : int or string
        Labels of feature axis.

        If int the corresponding number of labels are selected from the alphabet 'abcde...'

    join : None or dict
        Parameter combinations to join

        If a data set has parametrized by time t and stimulus s, then dPCA will split
        the data into marginalizations corresponding to 't', 's' and 'ts'. At times,
        we want to join different marginalizations (like 's' and 'ts'), e.g. if
        we are only interested in the time-modulated stimulus components. In this case,
        we would pass {'ts' : ['s','ts']}.

    regularizer : None, float, 'auto'
        Regularization parameter. If None or 0, then no regularization is applied.
        For float, the regularization weight is regularizer*var(data). If 'auto', the
        optimal regularization parameter is found during fitting (might take some time).

    n_components : None, int or dict
        Number of components to keep.

        If n_components is int, then the same number of components are kept in every
        marginalization. Otherwise, the dict allows to set the number of components
        in each marginalization (e.g. {'t' : 10, 'ts' : 5}). Defaults to 10.

    copy : bool
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    Attributes
    ----------
    explained_variance_ratio_ : dict with arrays, [n_components]
        Dictionary in which each key refers to one marginalization and the \
        value is a vector with the percentage of variance explained by each of \
        the marginal components.

    Notes
    -----
    Implements the dPCA model from:
    D Kobak*, W Brendel*, C Constantinidis, C Feierstein, A Kepecs, Z Mainen, \
    R Romo, X-L Qi, N Uchida, C Machens
    Demixed principal component analysis of population activity in higher \
    cortical areas reveals independent representation of task parameters,


    Examples
    --------

    >>> import numpy as np
    >>> from dPCA import dPCA
    >>> # X shape: (n_neurons, n_time_points)
    >>> X = np.random.randn(10, 100)
    >>> dpca = dPCA(labels='t', n_components=2)
    >>> dpca.fit(X)
    dPCA(n_components=2, ...)
    >>> print(dpca.explained_variance_ratio_['t'])
    [ 0.15...  0.05...]
    """
    
    def __init__(self, labels=None, join=None, n_components=10, regularizer=None, 
                 scale=False, noise_covariance_type='pooled', simultaneous=False, 
                 cv_method='training'):
        if isinstance(labels,str):
            self.labels = labels
        elif isinstance(labels,int):
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
            labels = alphabet[:labels]
        else:
            raise TypeError("""Wrong type for labels. Please either set labels to the number of
                               variables or provide the axis labels as a single string of characters 
                               (like "ts" for time and stimulus)""")

        self._join = join
        self.regularizer = 0 if regularizer == None else regularizer
        self.opt_regularizer_flag = regularizer == 'auto'
        self.n_components = n_components
        self.marginalizations = self._get_parameter_combinations()
        self.scale = scale
        self.noise_covariance_type = noise_covariance_type
        self.simultaneous = simultaneous
        self.cv_method = cv_method

        self.debug = 2
        
        if self.cv_method not in ['training', 'neuronwise']:
            raise ValueError('cv_method must be either "training" or "neuronwise"')

        self.n_trials = 5
        self.protect = None
        if regularizer == 'auto':
            print("""You chose to determine the regularization parameter automatically. This can
                    take substantial time and grows linearly with the number of crossvalidation
                    folds. The latter can be set by changing self.n_trials (default = 5). Similarly,
                    use self.protect to set the list of axes that are not supposed to get to get shuffled
                    (e.g. upon splitting the data into test- and training, time-points should always
                    be drawn from the same trial, i.e. self.protect = ['t']). This can significantly
                    speed up the code.""")

    def fit(self, X, trialX=None):
        self._fit(X, trialX=trialX)
        return self

    def _fit(self, X, trialX=None, mXs=None, center=True, SVD=None, optimize=True, Cnoise=None):
        """ Fit the model on X

        Parameters
        ----------
            X: array-like, shape (n_samples, n_features_1, n_features_2, ...)
                Training data, where n_samples in the number of samples
                and n_features_j is the number of the j-features (where the axis correspond
                to different parameters).

            trialX: array-like, shape (n_trials, n_samples, n_features_1, n_features_2, ...)
                Trial-by-trial data. Shape is similar to X but with an additional axis at the beginning
                with different trials. If different combinations of features have different number
                of trials, then set n_samples to the maximum number of trials and fill unoccupied data
                points with NaN.

            mXs: dict with values in the shape of X
                Marginalized data, should be the result of dpca._marginalize

            center: bool
                Centers data if center = True

            SVD: list of arrays
                Singular-value decomposition of the data. Don't provide!

            optimize: bool
                Flag to turn automatic optimization of regularization parameter on or off. Needed
                internally.
                
            Cnoise: array-like, shape (n_features, n_features)
                Noise covariance matrix to be added to the regularization term.
        """

        def flat2d(A):
            ''' Flattens all but the first axis of an ndarray, returns view. '''
            return A.reshape((A.shape[0],-1))

        n_features = X.shape[0]

        if center:
            X = X - np.mean(flat2d(X),1).reshape((n_features,) + len(self.labels)*(1,))

        if mXs is None:
            mXs = self._marginalize(X)

        # ----------------------------------------------------------------------
        # Compute and store Signal/Noise statistics if trialX is provided
        # ----------------------------------------------------------------------
        if trialX is not None:
            # 1. Compute Noise Covariance (Analytical Approach)
            N_samples = self._get_n_samples(trialX, protect=self.protect)
            self.Cnoise = get_noise_covariance(X, trialX, N_samples, 
                                             simultaneous=self.simultaneous, 
                                             type=self.noise_covariance_type)
            
            # 2. Compute effective trials (Ktilde) per neuron for scaling
            # N_samples shape: (n_features, ...conditions...) -> (n_features, -1) -> mean(axis=1)
            Ktilde = np.mean(N_samples.reshape(N_samples.shape[0], -1), axis=1)
            
            # 3. Total Statistics
            total_variance = np.sum(X**2)
            
            # Total Noise Variance in the trial-averaged data: sum(diag(self.Cnoise) / Ktilde)
            self.total_noise_var_ = np.sum(np.diag(self.Cnoise) / Ktilde)
            self.total_signal_var_ = total_variance - self.total_noise_var_
            
            # Pre-compute noise projection term: Cnoise_mean = Cnoise / sqrt(K_i * K_j)
            K_outer = np.sqrt(np.outer(Ktilde, Ktilde))
            self.Cnoise_mean_ = self.Cnoise / K_outer

        if self.opt_regularizer_flag and optimize:
            if self.debug > 0:
                print("Start optimizing regularization.")

            if trialX is None:
                raise ValueError('To optimize the regularization parameter, the trial-by-trial data trialX needs to be provided.')

            self._optimize_regularization(X,trialX)

        if Cnoise is None and trialX is not None:
            Cnoise = self.Cnoise

        try:
            if self.regularizer > 0:
                regX, regmXs, pregX = self._add_regularization(X,mXs,self.regularizer*np.sum(X**2),SVD=SVD,Cnoise=Cnoise)
            else:
                if Cnoise is not None:
                    regX, regmXs, pregX = self._add_regularization(X,mXs,0,SVD=SVD,Cnoise=Cnoise)
                else:
                    regX, regmXs, pregX = X, mXs, pinv(X.reshape((n_features,-1)))
        except np.linalg.LinAlgError:
            if self.debug > 0:
                print("Matrix close to singular, using tiny regularization, lambda = 1e-10")
            tiny_lam = 1e-10
            regX, regmXs, pregX = self._add_regularization(X,mXs,tiny_lam*np.sum(X**2),SVD=SVD,Cnoise=Cnoise)

        self.P, self.D = self._solve_dpca(regX,regmXs,pinvX=pregX)

        # ------------------------------------------------------------------
        # Compute Explained Variance
        # ------------------------------------------------------------------
        total_variance = np.sum(X**2)
        X_flat = X.reshape((X.shape[0], -1))

        def marginal_variances(marginal):
            ''' Computes the relative variance explained of each component
                within a marginalization
            '''
            D, P, Xr = self.D[marginal], self.P[marginal], X_flat

            return [np.sum(np.outer(P[:,k], np.dot(D[:,k], Xr))**2) / total_variance for k in range(D.shape[1])]

        self.explained_variance_ratio_ = {}
        for key in list(self.marginalizations.keys()):
            self.explained_variance_ratio_[key] = marginal_variances(key)
        
        # ------------------------------------------------------------------
        # PCA and Signal/Noise Variance Decomposition
        # ------------------------------------------------------------------
        
        # 1. PCA Explained Variance
        # Compute SVD of X to get optimal PCA variance
        # X_flat is (n_features, n_samples)
        U, s, Vh = np.linalg.svd(X_flat, full_matrices=False)
        self.explained_variance_ratio_pca_ = s**2 / total_variance

        # 2. Signal and Noise Variance (if available from fit)
        if hasattr(self, 'Cnoise_mean_') and hasattr(self, 'total_signal_var_'):
            self.explained_variance_ratio_noise_ = {}
            self.explained_variance_ratio_signal_ = {}
            self.marginalization_signal_ratios_ = {}
            
            for key in self.marginalizations.keys():
                if key in self.D:
                    D = self.D[key]
                    # Projected Noise Variance: diag(D^T * Cnoise_mean * D)
                    # We normalize by Total Data Variance to keep units consistent with explained_variance_ratio_
                    noise_vars_projected = np.diag(np.dot(D.T, np.dot(self.Cnoise_mean_, D)))
                    
                    ratio_noise = noise_vars_projected / total_variance
                    ratio_total = np.array(self.explained_variance_ratio_[key])
                    
                    # Signal = Total - Noise
                    ratio_signal = ratio_total - ratio_noise
                    
                    self.explained_variance_ratio_noise_[key] = ratio_noise
                    self.explained_variance_ratio_signal_[key] = ratio_signal
                    
                    # Ratio of Total Signal Variance attributed to this marginalization
                    # Sum of signal variance of components in this marginalization / Total Signal Variance
                    signal_var_captured = np.sum(ratio_signal) * total_variance
                    self.marginalization_signal_ratios_[key] = signal_var_captured / self.total_signal_var_     

    def fit_transform(self, X, trialX=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features_1, n_features_2, ...)
            Training data, where n_samples in the number of samples
            and n_features_j is the number of the j-features (where the axis correspond
            to different parameters).

        Returns
        -------
        X_new : dict with arrays with the same shape as X
            Dictionary in which each key refers to one marginalization and the value is the
            latent component.

        """
        self._fit(X,trialX=trialX)

        return self.transform(X)

    def _get_parameter_combinations(self, join=True):
        ''' Returns all parameter combinations, e.g. for labels = 'xyz'

            {'x' : (0,), 'y' : (1,), 'z' : (2,), 'xy' : (0,1), 'xz' : (0,2), 'yz' : (1,2), 'xyz' : (0,1,2)}

            If join == True, parameter combinations are condensed according to self._join, Otherwise all
            combinations are returned.
        '''
        # subsets = () (0,) (1,) (2,) (0,1) (0,2) (1,2) (0,1,2)"
        subsets = list(chain.from_iterable(combinations(list(range(len(self.labels))), r) for r in range(len(self.labels))))

        # delete empty set & add (0,1,2)
        del subsets[0]
        subsets.append(list(range(len(self.labels))))

        # create dictionary
        pcombs = OrderedDict()
        for subset in subsets:
            key = ''.join([self.labels[i] for i in subset])
            pcombs[key] = set(subset)

        # condense dict if not None
        if isinstance(self._join,dict) and join:
            for key, combs in self._join.items():
                tmp = [pcombs[comb] for comb in combs]

                for comb in combs:
                    del pcombs[comb]

                pcombs[key] = tmp

        return pcombs

    def _marginalize(self, X, save_memory=False):
        """ Marginalize the data matrix

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features_1, n_features_2, ...)
            Training data, where n_samples in the number of samples
            and n_features_j is the number of the j-features (where the axis correspond
            to different parameters).

        save_memory : bool, set to True if memory really is an issue (though optimization is not perfect yet)

        Returns
        -------
        mXs : dictionary, with values corresponding to the marginalized data (and the key refers to the marginalization)
        """

        def mmean(X,axes,expand=False):
            ''' Takes mean along several axis (given as list). If expand the averaged dimensions will be filled with
                new axis to retain the dimension.
            '''
            Z = X.copy()

            for ax in np.sort(axes)[::-1]:
                Z = np.mean(Z,ax)

                if expand == True:
                    Z = np.expand_dims(Z,ax)

            return Z

        def dense_marg(Y,mYs):
            ''' The original marginalizations as returned by "get_marginalizations" are sparse in the sense that
                marginalized axis are newaxis. This functions blows them up to the original size of the data set
                (need for optimization).
            '''
            tmp = np.zeros_like(Y)
            for key in list(mYs.keys()):

                mYs[key] = (tmp + mYs[key]).reshape((Y.shape[0],-1))

            return mYs

        Xres = X.copy()      # residual of data

        # center data
        Xres -= np.mean(Xres.reshape((Xres.shape[0],-1)),-1).reshape((Xres.shape[0],) + (len(Xres.shape)-1)*(1,))

        # init dict with marginals
        Xmargs = OrderedDict()

        # get parameter combinations
        pcombs = self._get_parameter_combinations(join=False)

        # subtract the mean
        S = list(pcombs.values())[-1]    # full set of indices

        if save_memory:
            for key, phi in pcombs.items():
                S_without_phi = list(S - phi)

                # compute marginalization and save
                Xmargs[key] = mmean(Xres,np.array(S_without_phi)+1,expand=True)

                # subtract the marginalization from the data
                Xres -= Xmargs[key]
        else:
            # efficient precomputation of means
            pre_mean = {}

            for key, phi in pcombs.items():
                if len(key) == 1:
                    pre_mean[key] = mmean(Xres,np.array(list(phi))+1,expand=True)
                else:
                    pre_mean[key] = mmean(pre_mean[key[:-1]],np.array([list(phi)[-1]])+1,expand=True)

            # compute marginalizations
            for key, phi in pcombs.items():
                key_without_phi = ''.join(filter(lambda ch: ch not in key, self.labels))
                # self.labels.translate(None, key)

                # build local dictionary for numexpr
                X = pre_mean[key_without_phi] if len(key_without_phi) > 0 else Xres

                if len(key) > 1:
                    subsets = list(chain.from_iterable(combinations(key, r) for r in range(1,len(key))))
                    subsets = [''.join(subset) for subset in subsets]
                    local_dict = {subset : Xmargs[subset] for subset in subsets}
                    local_dict['X'] = X

                    Xmargs[key] = ne.evaluate('X - ' + ' - '.join(subsets),local_dict=local_dict)
                else:
                    Xmargs[key] = X

        # condense dict if not None
        if isinstance(self._join,dict):
            for key, combs in self._join.items():
                Xshape = np.ones(len(self.labels)+1,dtype='int')
                for comb in combs:
                    sh = np.array(Xmargs[comb].shape)
                    Xshape[(sh-1).nonzero()] = sh[(sh-1).nonzero()]

                tmp = np.zeros(Xshape)

                for comb in combs:
                    tmp += Xmargs[comb]
                    del Xmargs[comb]

                Xmargs[key] = tmp

        Xmargs = dense_marg(X,Xmargs)

        return Xmargs

    def _optimize_regularization(self, X, trialX, center=True, lams='auto'):
        """ Optimization routine to find optimal regularization parameter.

            TO DO: Routine is pretty dumb right now (go through predetermined
            list and find minimum). There  are several ways to speed it up.
        """

        # center data
        if center:
            X = X - np.mean(X.reshape((X.shape[0],-1)),1).reshape((X.shape[0],)\
                  + len(self.labels)*(1,))

        # compute variance of data
        varX = np.sum(X**2)

        # test different inits and regularization parameters
        if lams == 'auto':
            N = 26
            lams = np.logspace(0,N,num=N, base=1.5, endpoint=False)*1e-7

        # compute crossvalidated score over n_trials repetitions
        scores = self.crossval_score(lams,X,trialX,mean=False)

        # take mean over total scores
        totalscore = np.mean(np.sum(np.dstack([scores[key] for key in list(scores.keys())]),-1),0)

        # Raise warning if optimal lambda lies at boundaries
        if np.argmin(totalscore) == 0 or np.argmin(totalscore) == len(totalscore) - 1:
            if self.debug > 0:
                print("Warning: Optimal regularization parameter lies at the \
                       boundary of the search interval. Please provide \
                       different search list (key: lams).")

        # set minimum as new lambda
        self.regularizer = lams[np.argmin(totalscore)]

        if self.debug > 1:
            print('Optimized regularization, optimal lambda = ', self.regularizer)
            print('Regularization will be fixed; to compute the optimal \
                   parameter again on the next fit, please \
                   set opt_regularizer_flag to True.')

            self.opt_regularizer_flag = False
    
    def crossval_score(self, lams, X, trialX, mean=True):
        """ Calculates crossvalidation scores for a given set of regularization
            parameters. To this end it takes one parameter off the list,
            computes the model on a training set and then validates the
            reconstruction performance on a validation set.

            Parameters
            ----------
            lams: 1D array of floats
                Array of regularization parameters to test.

            X: array-like, shape (n_samples, n_features_1, n_features_2, ...)
                Training data, where n_samples in the number of samples
                and n_features_j is the number of the j-features (where the
                axis correspond to different parameters).

            trialX: array-like, shape (n_trials, n_samples, n_features_1, n_features_2, ...)
                Trial-by-trial data. Shape is similar to X but with an additional axis at the beginning
                with different trials. If different combinations of features have different number
                of trials, then set n_samples to the maximum number of trials and fill unoccupied data
                points with NaN.

            mean: bool (default: True)
                Set True if the crossvalidation score should be averaged over
                all marginalizations, otherwise False.

            Returns
            -------
            mXs : dictionary, with values corresponding to the marginalized
                  data (and the key refers to the marginalization)
        """
        scores = np.zeros((self.n_trials,len(lams))) if mean else {key : np.zeros((self.n_trials,len(lams))) for key in list(self.marginalizations.keys())}

        N_samples = self._get_n_samples(trialX,protect=self.protect)

        for trial in range(self.n_trials):
            print("Starting trial ", trial + 1, "/", self.n_trials)

            trainX, validX = self.train_test_split(X,trialX,N_samples=N_samples)

            if self.noise_covariance_type != 'none':
                CnoiseTrain = get_noise_covariance(trainX, trialX, N_samples - 1, simultaneous=self.simultaneous, type=self.noise_covariance_type)
            else:
                CnoiseTrain = None

            trainmXs, validmXs = self._marginalize(trainX), self._marginalize(validX)
            
            # Select target marginals based on CV method
            if self.cv_method == 'training':
                target_mXs = trainmXs
            elif self.cv_method == 'neuronwise':
                target_mXs = validmXs

            for k, lam in enumerate(lams):
                self.regularizer = lam
                self._fit(trainX,mXs=trainmXs,Cnoise=CnoiseTrain,optimize=False)

                if mean:
                    scores[trial,k] = self._score(validX,target_mXs)
                else:
                    tmp = self._score(validX,target_mXs,mean=False)
                    for key in list(self.marginalizations.keys()):
                        scores[key][trial,k] = tmp[key]

        return scores    
    
    def _score(self, X, mXs, mean=True):
        """ Scoring for crossvalidation.
        
            If cv_method == 'neuronwise', it predicts one observable (e.g. one neuron) of X at a time, using all other dimensions:
            \sum_phi ||X[n] - F_\phi D_phi^{-n} X^{-n}||^2
            where phi refers to the marginalization and X^{-n} (D_phi^{-n}) are all rows of X (D) except the n-th row.
            
            If cv_method == 'training', it computes the reconstruction error of the target marginals (mXs) using the test data (X):
            \sum_phi ||mXs - F_\phi D_phi^T X||^2
        """
        n_features = X.shape[0]
        X = X.reshape((n_features,-1))

        error = {key: 0 for key in list(mXs.keys())}
        PDY  = {key : np.dot(self.P[key],np.dot(self.D[key].T,X)) for key in list(mXs.keys())}
        
        # Calculate diagonal terms only for neuronwise optimization
        if self.cv_method == 'neuronwise':
            trPD = {key : np.sum(self.P[key]*self.D[key],1) for key in list(mXs.keys())}

        for key in list(mXs.keys()):
            if self.cv_method == 'neuronwise':
                error[key] = np.sum((mXs[key] - PDY[key] + trPD[key][:,None]*X)**2)
            else:
                error[key] = np.sum((mXs[key] - PDY[key])**2)

        return error if not mean else np.sum(list(error.values()))

    def _solve_dpca(self, X, mXs, pinvX=None):
        """ Solves the dPCA minimization problem analytically by using exact eigenvalue decomposition.

        Returns
        -------
        P : dict mapping strings to array-like,
            Holds encoding matrices for each term in variance decompostions (used in inverse_transform
            to map from low-dimensional representation back to original data space).

        D : dict mapping strings to array-like,
            Holds decoding matrices for each term in variance decompostions (used to transform data
            to low-dimensional space).

        """
        from scipy.sparse.linalg import eigsh

        n_features = X.shape[0]
        rX = X.reshape((n_features,-1))
        pinvX = pinv(rX) if pinvX is None else pinvX

        P, D = {}, {}

        for key in list(mXs.keys()):
            mX = mXs[key].reshape((n_features,-1)) # called X_phi in paper
            C = np.dot(mX,pinvX)
            
            # Calculate covariance matrix for exact eigendecomposition
            M = np.dot(C, rX)
            MMT = np.dot(M, M.T)

            if isinstance(self.n_components,dict):
                n_comp = self.n_components[key]
            else:
                n_comp = self.n_components

            # Extract exact top n_comp eigenvectors
            evals, evecs = eigsh(MMT, k=n_comp, which='LM')
            
            # eigsh returns eigenvalues in ascending order, reverse to match descending variance
            idx = np.argsort(evals)[::-1]
            U = evecs[:, idx]

            # flip axes such that all encoders have more positive values
            for i in range(U.shape[1]):
                if np.sum(np.sign(U[:,i])) < 0:
                    U[:,i] *= -1

            P[key] = U
            D[key] = np.dot(U.T,C).T

            if self.scale:
                for i in range(D[key].shape[1]):
                    B = np.outer(P[key][:, i], np.dot(D[key][:, i].T, rX))
                    scaling_factor = np.sum(mX * B) / np.sum(B * B)
                    D[key][:, i] *= scaling_factor

        return P, D    
    
    def _add_regularization(self,Y,mYs,lam,SVD=None,pre_reg=False,Cnoise=None):
        """ Prepares the data matrix and its marginalizations for the randomized_dpca solver (see paper)."""
        n_features = Y.shape[0]

        if not pre_reg:
            regY = np.hstack([Y.reshape((n_features,-1)),lam*np.eye(n_features)])
        else:
            regY = Y
            regY[:,-n_features:] = lam*np.eye(n_features)

        if not pre_reg:
            regmYs = OrderedDict()

            for key in list(mYs.keys()):
                regmYs[key] = np.hstack([mYs[key],np.zeros((n_features,n_features))])
        else:
            regmYs = mYs

        if SVD is not None and Cnoise is None:
            U,s,V = SVD

            M = ((s**2 + lam**2)**-1)[:,None]*U.T
            pregY = np.dot(np.vstack([V.T*s[None,:],lam*U]),M)
        else:
            covariance_term = np.dot(Y.reshape((n_features,-1)),Y.reshape((n_features,-1)).T) + lam**2*np.eye(n_features)
            if Cnoise is not None:
                covariance_term += Cnoise
            pregY = np.dot(regY.reshape((n_features,-1)).T,np.linalg.inv(covariance_term))

        return regY, regmYs, pregY
       
    def _zero_mean(self, X):
        """ Subtracts the mean from each observable """
        return X - np.mean(X.reshape((X.shape[0],-1)),1).reshape((X.shape[0],) + (len(X.shape)-1)*(1,))

    def _roll_back(self, X, axes, invert=False):
        ''' Rolls all axis in list crossval_protect to the end (or inverts if invert=True) '''
        rX = X
        axes = np.sort(axes)

        if invert:
            for ax in reversed(axes):
                rX = np.rollaxis(rX,-1,start=ax)
        else:
            for ax in axes:
                rX = np.rollaxis(rX,ax,start=len(X.shape))

        return rX

    def _get_n_samples(self, trialX, protect=None):
        """ Computes the number of samples for each parameter combinations (except along protect) """
        n_unprotect = len(trialX.shape) - len(protect) - 1 if protect is not None else len(trialX.shape) - 1
        n_protect   = len(protect) if protect is not None else 0

        return trialX.shape[0] - np.sum(np.isnan(trialX[(np.s_[:],) + (np.s_[:],)*n_unprotect + (0,)*n_protect]),0)

    def _check_protected(self, X, protect):
        ''' Checks if protect == None or, alternatively, if all protected axis are at the end '''
        if protect is None:
            protected = True
        else:
            # convert label in index
            protect = [self.labels.index(ax) for ax in protect]
            if set(protect) == set(np.arange(len(self.labels)-len(protect),len(self.labels))):
                protected = True
            else:
                protected = False
                print('Not all protected axis are at the end! While the algorithm will still work, the performance of the shuffling algorithm will substantially decrease due to unavoidable copies.')

        return protected

    def train_test_split(self, X, trialX, N_samples=None, sample_ax=0):
        """ Splits data in training and validation trial. To this end, we select one data-point in each observable for every
            combination of parameters (except along protected axis) for the validation set and average the remaining trial-by-trial
            data to get the training set.

            Parameters
            ----------
                X: array-like, shape (n_samples, n_features_1, n_features_2, ...)
                    Training data, where n_samples in the number of samples
                    and n_features_j is the number of the j-features (where the axis correspond
                    to different parameters).

                trialX: array-like, shape (n_trials, n_samples, n_features_1, n_features_2, ...)
                    Trial-by-trial data. Shape is similar to X but with an additional axis at the beginning
                    with different trials. If different combinations of features have different number
                    of trials, then set n_samples to the maximum number of trials and fill unoccupied data
                    points with NaN.

                N_samples: array-like with the same shape as X (except for protected axis).
                    Number of trials in each condition. If None, computed from trial data.


            Returns
            -------
                trainX: array-like, same shape as X
                    Training data

                blindX: array-like, same shape as X
                    Validation data

        """
        def flat2d(A):
            ''' Flattens all but the first axis of an ndarray, returns view. '''
            return A.reshape((A.shape[0],-1))

        protect = self.protect

        n_samples   = trialX.shape[-1]                       # number of samples
        n_unprotect = len(X.shape) - len(protect) if protect is not None else len(X.shape)
        n_protect   = len(protect) if protect is not None else 0

        if sample_ax != 0:
            raise NotImplemented('The sample axis needs to come first.')

        # test if all protected axes lie at the end
        protected = self._check_protected(trialX,protect)

        # reorder matrix to protect certain axis (for speedup)
        if ~protected:
            # turn crossval_protect into index listX
            axes = [self.labels.index(ax) + 2 for ax in protect]

            # reorder matrix
            trialX = self._roll_back(trialX,axes)
            X = np.squeeze(self._roll_back(X[None,...],axes))

        # compute number of samples in each condition
        if N_samples is None:
            N_samples = self._get_n_samples(trialX,protect=self.protect)

        # get random indices
        idx = (np.random.rand(*N_samples.shape)*N_samples).astype(int)

        # select values
        blindX = np.empty(trialX.shape[1:])

        # iterate over multi_index
        it = np.nditer(np.empty(N_samples.shape), flags=['multi_index'])

        while not it.finished:
            blindX[it.multi_index + (np.s_[:],)*n_protect] = trialX[(idx[it.multi_index],) + it.multi_index + (np.s_[:],)*n_protect]
            it.iternext()

        # compute trainX
        trainX = (X*(N_samples/(N_samples-1))[(np.s_[:],)*n_unprotect + (None,)*n_protect] - blindX/(N_samples-1)[(np.s_[:],)*n_unprotect + (None,)*n_protect])

        # inverse rolled axis in blindX
        if ~protected:
            blindX = self._roll_back(blindX[...,None],axes,invert=True)[...,0]
            trainX = self._roll_back(trainX[...,None],axes,invert=True)[...,0]

        # remean datasets (both equally)
        trainX -= np.mean(flat2d(trainX),1)[(np.s_[:],) + (None,)*(len(X.shape)-1)]
        blindX -= np.mean(flat2d(blindX),1)[(np.s_[:],) + (None,)*(len(X.shape)-1)]

        return trainX, blindX

    def shuffle_labels(self, trialX):
        """ Shuffles *inplace* labels between conditions in trial-by-trial data, respecting the number of trials per condition.

            Parameters
            ----------
                trialX: array-like, shape (n_trials, n_samples, n_features_1, n_features_2, ...)
                    Trial-by-trial data. Shape is similar to X but with an additional axis at the beginning
                    with different trials. If different combinations of features have different number
                    of trials, then set n_samples to the maximum number of trials and fill unoccupied data
                    points with NaN.

        """

        # import shuffling algorithm from cython source
        protect = self.protect

        # test if all protected axes lie at the end
        protected = self._check_protected(trialX,protect)

        # reorder matrix to protect certain axis (for speedup)
        if ~protected:
            # turn crossval_protect into index list
            axes = [self.labels.index(ax) + 2 for ax in protect]

            # reorder matrix
            trialX = self._roll_back(trialX,axes)

        # reshape all non-protect axis into one vector
        original_shape = trialX.shape
        trialX = trialX.reshape((-1,) + trialX.shape[-len(protect):])

        # reshape all protected axis into one
        original_shape_protected = trialX.shape
        trialX = trialX.reshape((trialX.shape[0],-1))

        # shuffle within non-protected axis
        shuffle2D(trialX)

        # inverse reshaping of protected axis
        trialX = trialX.reshape(original_shape_protected)

        # inverse reshaping & sample axis
        trialX = trialX.reshape(original_shape)
        #trialX = np.rollaxis(trialX,0,len(original_shape))

        # inverse rolled axis in trialX
        if protected:
            trialX = self._roll_back(trialX,axes,invert=True)

        return trialX

    def significance_analysis(self, X, trialX, n_shuffles=100, n_splits=100,n_consecutive=10, 
                              n_eval_comps=3,axis=None,full=False):
        '''
            Cross-validated significance analysis of dPCA model. Here the generalization from the training
            to test data is tested by a simple classification measure in which one tries to predict the
            label of a validation test point from the training data. The performance is tested for n_splits
            test and training separations. The classification performance is then compared against
            the performance on data with randomly shuffled labels. Only if the performance is higher
            then the maximum in the shuffled data we regard the component as significant.

            Parameters
            ----------
                X: array-like, shape (n_samples, n_features_1, n_features_2, ...)
                    Training data, where n_samples in the number of samples
                    and n_features_j is the number of the j-features (where the axis correspond
                    to different parameters).

                trialX: array-like, shape (n_trials, n_samples, n_features_1, n_features_2, ...)
                    Trial-by-trial data. Shape is similar to X but with an additional axis at the beginning
                    with different trials. If different combinations of features have different number
                    of trials, then set n_samples to the maximum number of trials and fill unoccupied data
                    points with NaN.

                n_shuffles: integer
                    Number of label shuffles over which the maximum is taken (default = 100, which
                    is equivalent to p > 0.01)

                n_splits: integer
                    Number of train-test splits per shuffle, from which the average performance is
                    deduced.

                n_consecutive: integer
                    Sometimes individual data points are deemed significant purely by chance. To reduced
                    such noise one can demand that at least n consecutive data points are rated as significant.

                n_eval_comps: integer
                    Number of components to evaluate for classification accuracy (default = 3).
                    
                axis: None or True (default = None)
                    Determines whether the significance is calculated over the last axis. More precisely,
                    one is often interested in determining the significance of a component over time. In this
                    case, set axis to True and make sure the last axis is time.

                full: Boolean (default = False)
                    Whether or not all scores are returned. If False, only the significance matrix is returned.


            Returns
            -------
                masks: Dictionary
                    Dictionary with keys corresponding to the marginalizations and with values that are
                    binary nparrays that capture the significance of the demixed components.

                true_score: Dictionary  (only returned when full = True)
                    Dictionary with the scores of the data.

                scores: Dictionary  (only returned when full = True)
                    Dictionary with the scores of the shuffled data.

        '''
        explained_variance_ratio_backup = getattr(self, 'explained_variance_ratio_', None)
        had_evr = hasattr(self, 'explained_variance_ratio_')

        full_P = None
        full_D = None
        full_regularizer = None

        try:
            assert axis in [None, True]
            
            if self.opt_regularizer_flag:
                print("Regularization not optimized yet; start optimization now.")
                self._optimize_regularization(X,trialX)

            self.fit(X)
            
            full_D = {key: self.D[key].copy() for key in self.D.keys()}
            full_P = {key: self.P[key].copy() for key in self.P.keys()}
            full_regularizer = self.regularizer
            
            keys = list(self.marginalizations.keys())
            if self.labels[-1] in keys:
                keys.remove(self.labels[-1])
            
            def compute_mean_score(X,trialX,n_splits):
                K = 1 if axis is None else X.shape[-1]
                data_ndim = len(X.shape) - 1

                if type(self.n_components) == int:
                    scores = {key : np.empty((min(n_eval_comps, self.n_components), n_splits, K)) for key in keys}
                else:
                    scores = {key : np.empty((min(n_eval_comps, self.n_components[key]), n_splits, K)) for key in keys}

                for shuffle in range(n_splits):
                    print('.', end=' ')

                    trainX, validX = self.train_test_split(X,trialX)

                    trainZ = self.fit_transform(trainX)
                    validZ = self.transform(validX)

                    for key in keys:
                        ncomps = self.n_components if type(self.n_components) == int else self.n_components[key]
                        ncomps_eval = min(n_eval_comps, ncomps)

                        dots = np.dot(full_D[key].T, self.D[key])
                        aligned_trainZ = np.zeros_like(trainZ[key])
                        aligned_validZ = np.zeros_like(validZ[key])
                        
                        for c in range(ncomps_eval):
                            best_match = np.argmax(np.abs(dots[c, :]))
                            sign = np.sign(dots[c, best_match])
                            if sign == 0: sign = 1
                            aligned_trainZ[c] = trainZ[key][best_match] * sign
                            aligned_validZ[c] = validZ[key][best_match] * sign
                            dots[:, best_match] = 0
                        
                        axset = self.marginalizations[key]
                        axset = axset if type(axset) == set else set.union(*axset)
                        
                        key_axes = list(axset)
                        non_key_axes = [ax for ax in range(data_ndim - (1 if axis is not None else 0)) if ax not in key_axes]
                        time_axis = [data_ndim - 1] if axis is not None else []
                        
                        transpose_order = key_axes + non_key_axes + time_axis
                        
                        for comp in range(ncomps_eval):
                            train_comp = aligned_trainZ[comp]
                            valid_comp = aligned_validZ[comp]
                            
                            train_transposed = np.transpose(train_comp, transpose_order)
                            valid_transposed = np.transpose(valid_comp, transpose_order)
                            
                            Q = np.prod([train_comp.shape[i] for i in key_axes], dtype=int) if key_axes else 1
                            M = np.prod([train_comp.shape[i] for i in non_key_axes], dtype=int) if non_key_axes else 1
                            T_dim = train_comp.shape[time_axis[0]] if time_axis else 1
                            
                            train_reshaped = train_transposed.reshape((Q, M, T_dim))
                            valid_reshaped = valid_transposed.reshape((Q, M, T_dim))
                            
                            accuracy = nearest_centroid_accuracy(train_reshaped, valid_reshaped)
                            
                            scores[key][comp, shuffle] = accuracy

                for key in keys:
                    scores[key] = np.nanmean(scores[key], axis=1)
                            
                return scores

            trialX = trialX.copy()

            print("Compute score of data: ", end=' ')
            true_score = compute_mean_score(X,trialX,n_splits)
            print("Finished.")

            scores = {key : [] for key in keys}

            for it in range(n_shuffles):
                print("\rCompute score of shuffled data: ", str(it), "/", str(n_shuffles), end=' ')

                self.shuffle_labels(trialX)

                X_shuffled = np.nanmean(trialX,axis=0)

                score = compute_mean_score(X_shuffled,trialX,n_splits)

                for key in keys:
                    scores[key].append(score[key])

            masks = {}
            for key in keys:
                maxscore = np.amax(np.dstack(scores[key]),-1)
                masks[key] = true_score[key] > maxscore

            if n_consecutive > 1:
                for key in keys:
                    mask = masks[key]

                    for k in range(mask.shape[0]):
                        masks[key][k,:] = denoise_mask(masks[key][k].astype(np.int32),n_consecutive)

            if full:
                return masks, true_score, scores
            else:
                return masks
        finally:
            if had_evr:
                self.explained_variance_ratio_ = explained_variance_ratio_backup
            elif hasattr(self, 'explained_variance_ratio_'):
                del self.explained_variance_ratio_

            if full_D is not None:
                self.D = full_D
            if full_P is not None:
                self.P = full_P
            if full_regularizer is not None:
                self.regularizer = full_regularizer    
    
    def transform(self, X, marginalization=None):
        """Apply the dimensionality reduction on X.

        X is projected on the first principal components previous extracted
        from a training set.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features_1, n_features_2, ...)
            Training data, where n_samples in the number of samples
            and n_features_j is the number of the j-features (where the axis correspond
            to different parameters).

        marginalization : str or None
            Marginalization subspace upon which to project, if None return dict
            with projections on all marginalizations

        Returns
        -------
        X_new : dict with arrays of the same shape as X
            Dictionary in which each key refers to one marginalization and the value is the
            latent component. If specific marginalization is given, returns only array

        """
        X = self._zero_mean(X)
        X_flat = X.reshape((X.shape[0], -1))

        if marginalization is not None:
            D, Xr         = self.D[marginalization], X_flat
            X_transformed = np.dot(D.T, Xr).reshape((D.shape[1],) + X.shape[1:])
        else:
            X_transformed = {}
            for key in list(self.marginalizations.keys()):
                X_transformed[key] = np.dot(self.D[key].T, X_flat).reshape((self.D[key].shape[1],) + X.shape[1:])

        return X_transformed
    
    def inverse_transform(self, X, marginalization=None):
        """ Transform data back to its original space, i.e.,
        return an input X_original whose transform would be X

        Parameters
        ----------
        X : array-like or dict
            New data, where n_samples is the number of samples
            and n_components is the number of components.
            If marginalization is None, X must be a dictionary mapping 
            marginalization keys to component arrays.

        marginalization : str or None (default: None)
            The marginalization to reconstruct. If None, reconstructs from all
            provided in X (which must be a dict).

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)
        """
        if marginalization is None:
            if not isinstance(X, dict):
                raise TypeError("When marginalization is None, X must be a dictionary.")
            
            X_original = None
            
            for key, val in X.items():
                # Recursively reconstruct each marginalization
                part = self.inverse_transform(val, marginalization=key)
                
                if X_original is None:
                    X_original = part
                else:
                    X_original += part
            
            return X_original
        else:
            X = self._zero_mean(X)
            X_transformed = np.dot(self.P[marginalization],X.reshape((X.shape[0],-1))).reshape((self.P[marginalization].shape[0],) + X.shape[1:])

            return X_transformed
        
    def reconstruct(self, X, marginalization):
        """ Transform data first into reduced space before projecting
        it back into data space. Equivalent to inverse_transform(transform(X)).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)
        """
        return self.inverse_transform(self.transform(X,marginalization),marginalization)
