import numpy as np

from sklearn.model_selection import train_test_split, KFold

from qml.kernels import gaussian_kernel, laplacian_kernel
from qml.kernels import gaussian_kernel_symmetric, laplacian_kernel_symmetric
from qml.math import cho_solve, cho_invert

from scipy.linalg import lstsq

import sys
import os
from datetime import datetime
from uuid import uuid3
from copy import copy

from memory_profiler import profile


class QML_KRR:

    """
    Class for performing basic QML operations with KRR.
    """


    def __init__(self):

        self.sigma = None
        self.lam = None
        self.cond = None
        self.method = None
        self.seed = None
        self.ktype = None

        self.params = None
        
        self.X = None
        self.X_train = None
        self.X_test = None
        
        self.Y = None
        self.Y_train = None
        self.Y_test = None

        self.kernel = None
        self.kernel_train = None
        self.kernel_pred = None

        self.train_ind = None
        self.test_ind = None

        self.n_sampels = None
        self.n_train = None
        self.n_test = None

        self.alpha = None

        self.Y_pred_train = None
        self.Y_pred_test = None

        self.mae_train = None
        self.mae_test = None
        self.mse_train = None
        self.mse_test = None

        return


    def add_params(self, **params):
        """
        Add parameters for the operations. If no paramters are given, 
        default values are chosen.
        sigma : 1000
        lam : 1e-12
        cond : 1e-9
        method : 'cho'
        ktype : 'laplacian'
        seed : 42
        
        args:
            ** params
        """
        
        if 'sigma' in params:
            self.sigma = params['sigma']
        else:
            self.sigma = 1000

        if 'lam' in params:
            self.lam = params['lam']
        else:
            self.lam = 1e-12

        if 'cond' in params:
            self.cond = params['cond']
        else:
            self.cond = 1e-9

        if 'method' in params:
            self.method = params['method']
        else:
            self.method = 'cho'

        if 'seed' in params:
            self.seed = params['seed']
        else:
            self.seed = 42

        if 'ktype' in params:
            self.ktype = params['ktype']
        else:
            self.ktype = 'laplacian'

        return


    def set_sigma(self, sigma):
        """
        Set kernel width.
        
        args:
            sigma : kernel width
        """

        self.sigma = sigma

        return


    def set_lam(self, lam):
        """
        Set regularization parameter:
        
        args:
            lam : regularization paramter
        """

        self.lam = lam

        return


    def set_method(self, method):
        """
        Chose method for the inversion of the kernel matrix ('cho', 'svd')

        args:
            method : method of choice
        """

        self.method = method

        return


    def set_cond(self, cond):
        """
        Set threshold for the minimal ratio of highest to lowest singular values
        if 'svd' has been chosen as method.

        args:
            cond : threshold
        """

        self.cond = cond

        return


    def set_seed(self, seed):
        """
        Set seed to start random choice of training and test set molecules.

        args:
            seed : seed to start the RNG
        """

        self.seed = seed

        return


    def set_ktype(self, ktype):
        """
        Set the type of kernel matrix desired ('laplacian', 'gaussian')

        args:
            ktype : type of kernel matrix
        """

        self.ktype = ktype

        return


    def get_params(self):
        """
        Gets back the parameters of this instance:

        returns: 
            params (dict)
        """

        self.params = {'sigma': self.sigma,
                       'lam': self.lam,
                       'cond': self.cond,
                       'method': self.method,
                       'seed': self.seed,
                       'ktype': self.ktype}

        return self.params


    def add_input(self, X):
        """
        Add input values to the object.
   
        args:
            X (2d-ndarray): inpute vector with representations
        """

        self.X = X
        self.n_samples = self.X.shape[0]

        return


    def add_labels(self, Y):
        """
        Add labels corresponding to the input values:
        
        args:
            Y (1d ndarray): labels corresponding to each input representation
        """

        self.Y = Y
        self.n_samples = self.Y.shape[0]

        return


    def get_train_test_indices(self, n_train):
        """
        Get the indices of the trianing and test set molecules:

        args:
            n_train (int): number of training set molecules
        
        returns:
            train_ind (1d ndarray): training set indices
            test_ind (1d ndarray) : test set indices
        """

        assert self.X is not None, 'input X not defined!'
        assert self.Y is not None, 'labels Y not defined!'
        assert self.X.shape[0] == self.Y.shape[0], 'input and labels have different lengths!'

        self.n_train = n_train
        self.n_test = self.n_samples - self.n_train

        self.train_ind, self.test_ind = train_test_split(np.arange(self.n_samples).astype(int),
                                                         train_size=self.n_train,
                                                         random_state=self.seed)

        return (self.train_ind, self.test_ind)


    def add_train_test_indices(self, train_ind, test_ind):
        """
        Add the training and test set indices.

        args: 
            train_ind (1d ndarray): training set indices
            test_ind (1d ndarray): test set indices
        """

        self.train_ind = train_ind
        self.test_ind = test_ind

        return


    def split_input(self):
        """
        Split input representations into training and test set;

        returns:
            X_train (2d ndarray): training set input representations
            X_test (2d ndarray): test set input representations
        """

        self.X_train = self.X[self.train_ind]
        self.X_test = self.X[self.test_ind]

        return (self.X_train, self.X_test)


    def split_labels(self):
        """
        Split labels into training and test set;

        returns:
            Y_train (1d ndarray): training set labels
            Y_test (1d ndarray): test set labels
        """

        self.Y_train = self.Y[self.train_ind]
        self.Y_test = self.Y[self.test_ind]

        return (self.Y_train, self.Y_test)


    def add_kernel(self, kernel):
        """
        Add kernel matrix:

        args: 
            kernel (2d ndarray) : add the kernel matrix
        """

        self.kernel = kernel
        self.n_samples = self.kernel.shape[0]

        return


    def split_kernel(self):
        """
        Split the kernel matrix  into a matrix for the training and predictions.

        returns:
            kernel_train (2d ndarray): kernel matrix for the regression
            kernel_pred (2d ndarray): kernel matrix used for the prediction
        """

        self.kernel_train = self.kernel[self.train_ind[:, np.newaxis], self.train_ind[np.newaxis]]
        self.kernel_pred = self.kernel[self.test_ind[:, np.newaxis], self.train_ind[np.newaxis]]

        return (self.kernel_train, self.kernel_pred)


    def add_kernel_train(self, kernel_train):
        """
        Add the training kernel matrix.
        
        args:
            kernel_train (2d ndarray): training kernel matrix
        """
        
        self.kernel_train = kernel_train

        return


    def add_kernel_pred(self, kernel_pred):
        """
        Add the prediction kernel matrix

        args:
            kernel_pred (2d ndarray): prediction kernel matrix
        """

        self.kernel_pred = kernel_pred

        return


    def compute_kernel(self):
        """
        Coompute the kernel matrix from the input representations and paramters given.

        returns:
            kernel: kernel matrix
        """

        assert self.X is not None, 'input X is not defined!'
        assert self.sigma is not None, 'kernel width sigma is not defined!'
        assert self.ktype is not None, 'kernel type (ktype) is not defined!'

        if self.ktype == 'laplacian':
            self.kernel = laplacian_kernel_symmetric(self.X, self.sigma)
        elif self.ktype == 'gaussian':
            self.kernel = gaussian_kernel_symmetric(self.X, self.sigma)

        return self.kernel


    def compute_kernel_train(self):
        """
        Compute the training kernel matrix from the input training representations 
        and paramters given.

        returns:
            kernel_train: training kernel matrix
        """

        assert self.X is not None, 'input X is not defined!'
        assert self.sigma is not None, 'kernel width sigma is not defined!'
        assert self.ktype is not None, 'kernel type (ktype) is not defined!'
        assert self.X_train is not None, 'X_train is not defined!'
        
        if self.ktype == 'laplacian':
            self.kernel_train = laplacian_kernel_symmetric(self.X_train, self.sigma)
        elif self.ktype == 'gaussian':
            self.kernel_train = gaussian_kernel_symmetric(self.X_train, self.sigma)

        return self.kernel_train
    

    def compute_kernel_pred(self):
        """
        Compute the prediction kernel matrix from the input training and test representations 
        and paramters given.

        returns:
            kernel_pred: prediction kernel matrix
        """

        assert self.X is not None, 'input X is not defined!'
        assert self.sigma is not None, 'kernel width sigma is not defined!'
        assert self.ktype is not None, 'kernel type (ktype) is not defined!'
        assert self.X_train is not None, 'X_train is not defined!'
        assert self.X_test is not None, 'X_test is not defined'

        if self.ktype == 'laplacian':
            self.kernel_pred = laplacian_kernel(self.X_test, self.X_train, self.sigma)
        elif self.ktype == 'gaussian':
            self.kernel_pred = gaussian_kernel(self.X_test, self.X_train, self.sigma)

        return self.kernel_pred


    def fit(self):
        """
        Fit the regression coefficients alpha given the training data.

        returns:
            alpha (1d ndarray): regression coefficients
        """

        if self.method == 'cho':
            self.alpha = cho_solve(self.kernel_train + self.lam * np.eye(self.kernel_train.shape[0]),
                                   self.Y_train)
        elif self.method == 'svd':
            self.alpha, res, r, s = lstsq(self.kernel_train, self.Y_train, cond=self.cond)

        return self.alpha


    def predict(self):
        """
        Predict the labels for the training and test set molecules:
        
        returns:
            Y_pred_train (1d ndarray): predictions on the training set
            Y_pred_test (1d ndarray): predictions on the test set
        """

        self.Y_pred_train = np.dot(self.kernel_train, self.alpha)
        self.Y_pred_test = np.dot(self.kernel_pred, self.alpha)

        return (self.Y_pred_train, self.Y_pred_test)


    def evaluate(self):
        """
        Evaluate results of the prediction:
        
        returns: 
            mae_train : mean absolute error on the training data
            mae_test : mean absolute error on the test data
        """

        res_train = self.Y_pred_train - self.Y_train
        res_test = self.Y_pred_test - self.Y_test

        self.mae_train = np.mean(np.abs(res_train))
        self.mae_test = np.mean(np.abs(res_test))

        self.mse_train = np.mean(res_train)
        self.mse_test = np.mean(res_test)

        return (self.mae_train, self.mae_test)


    def to_dict(self):

        return self.__dict__


    def get_summary(self):
        """
        Get a summary of the results.

        returns:
            summary (dict)
        """

        dt = datetime.now()
        date = dt.strftime('%Y%m%d%H%M%S')

        self.summary = {'sigma': self.sigma,
                        'lam': self.lam,
                        'cond': self.cond,
                        'method': self.method,
                        'seed': self.seed,
                        'ktype': self.ktype,
                        'n_samples': self.n_samples,
                        'n_train': self.n_train,
                        'n_test': self.n_test,
                        'train_ind': self.train_ind,
                        'test_ind': self.test_ind,
                        'Y_train': self.Y_train,
                        'Y_test': self.Y_test,
                        'Y_pred_train': self.Y_pred_train,
                        'Y_pred_test': self.Y_pred_test,
                        'mae_train': self.mae_train,
                        'mae_test': self.mae_test,
                        'mse_train': self.mse_test,
                        'mse_test': self.mse_test,
                        'date': date}

        return self.summary


class Delta_QML_KRR(QML_KRR):


    def __init__(self):

        super().__init__()

        self.Y_base = None
        self.Y_base_train = None
        self.Y_base_test = None

        self.d_train = None
        self.d_test = None
        self.d_pred_train = None
        self.d_pred_test = None

        self.mae_d_train = None
        self.mae_d_test = None
        self.mse_d_train = None
        self.mse_d_test = None

        return


    def add_base_labels(self, Y_base):
        """
        Add baseline labels corresponding to the input values:
        
        args:
            Y_base (1d ndarray): baseline labels corresponding to each input representation
        """

        self.Y_base = Y_base

        return


    def split_labels(self):
        """
        Split baseline labels into training and test set;

        returns:
            Y_train (1d ndarray): training set labels
            Y_test (1d ndarray): test set labels
        """

        self.Y_train, self.Y_test = super().split_labels()
        self.Y_base_train = self.Y_base[self.train_ind]
        self.Y_base_test = self.Y_base[self.test_ind]

        return (self.Y_train, self.Y_test)


    def fit(self):
        """
        Fit the regression coefficients alpha given the training data.

        returns:
            alpha (1d ndarray): regression coefficients
        """

        self.d_train = self.Y_train - self.Y_base_train
        self.d_test = self.Y_test - self.Y_base_test
        
        if self.method == 'cho':
            self.alpha = cho_solve(self.kernel_train + self.lam * np.eye(self.kernel_train.shape[0]),
                                   self.d_train)
        elif self.method == 'svd':
            self.alpha, res, r, s = lstsq(self.kernel_train, self.d_train, cond=self.cond)

        return self.alpha


    def predict(self):
        """
        Predict the labels for the training and test set molecules:
        
        returns:
            Y_pred_train (1d ndarray): predictions on the training set
            Y_pred_test (1d ndarray): predictions on the test set
        """

        self.d_pred_train = np.dot(self.kernel_train, self.alpha)
        self.d_pred_test = np.dot(self.kernel_pred, self.alpha)

        self.Y_pred_train = self.Y_base_train + self.d_pred_train
        self.Y_pred_test = self.Y_base_test + self.d_pred_test

        return (self.Y_pred_train, self.Y_pred_test)


    def evaluate(self):
        """
        Evaluate results of the prediction:
        
        returns: 
            mae_train : mean absolute error on the training data
            mae_test : mean absolute error on the test data
        """

        self.mae_train, self.mae_test = super().evaluate()

        d_res_train = self.d_pred_train - self.d_train
        d_res_test = self.d_pred_test - self.d_test

        self.mae_d_train = np.mean(np.abs(d_res_train))
        self.mae_d_test = np.mean(np.abs(d_res_test))

        self.mse_d_train = np.mean(d_res_train)
        self.mse_d_test = np.mean(d_res_test)

        return (self.mae_train, self.mae_test)


    def get_summary(self):
        """
        Get a summary of the results.

        returns:
            summary (dict)
        """

        self.summary = super().get_summary()

        self.summary['Y_base_train'] = self.Y_base_train
        self.summary['Y_base_test'] = self.Y_base_test
        self.summary['d_train'] = self.d_train
        self.summary['d_test']= self.d_test
        self.summary['d_pred_train'] = self.d_pred_train
        self.summary['d_pred_test'] = self.d_pred_test
        self.summary['mae_d_train'] = self.mae_d_train
        self.summary['mae_d_test'] = self.mae_d_test
        self.summary['mse_d_train'] = self.mse_d_train
        self.summary['mse_d_test'] = self.mse_d_test

        return self.summary
    


class CV_QML_KRR:

    """
    Class for cross-validation with QML and KRR.
    """

    def __init__(self, qml_krr):
        """
        args: 
            qml_krr : QML_KRR object
        """

        self.qml_krr = qml_krr

        self.n_fold = None

        self.mae_train_cv = None
        self.mae_test_cv = None
        self.std_train_cv = None
        self.std_test_cv = None

        return


    def add_params(self, **params):
        """
        Add parameters for cross-validation.
        default values:
        n_fold : 5

        args:
            ** params : CV parameters
        """

        if 'n_fold' not in params:
            self.n_fold = params['n_fold']
        else:
            self.n_fold = 5

        return


    def set_n_fold(self, n_fold):
        """
        Add number of folds for the CV.

        args:
            n_fold (int): number of folds
        """

        self.n_fold = n_fold

        return


    def run(self):
        """
        Iterate over CV folds.

        returns:
            results (dict)
        """

        kfold = KFold(self.n_fold, shuffle=True, random_state=self.qml_krr.seed)

        results = []

        for i, (train_ind, test_ind) in enumerate(kfold.split(np.arange(self.qml_krr.n_samples))):

            # split into training and test set
            
            self.qml_krr.add_train_test_indices(train_ind, test_ind)
            Y_train, Y_test = self.qml_krr.split_labels()
            kernel_train, kernel_pred = self.qml_krr.split_kernel()

            # actual regression starts here
            
            alpha = self.qml_krr.fit()
            Y_pred_train, Y_pred_test = self.qml_krr.predict()
            mae_train, mae_test = self.qml_krr.evaluate()

            # save the results
            
            results.append(copy(self.qml_krr.get_summary()))

        self.results = results

        return self.results


    def evaluate(self):
        """
        Evaluate CV results

        returns:
            mae_train_cv : mean absolute training error from CV
            mae_test_cv : mean absolute test error from CV
        """

        mae_train_tmp = []
        mae_test_tmp = []
        
        for i, res in enumerate(self.results):
            mae_train_tmp.append(res['mae_train'])
            mae_test_tmp.append(res['mae_test'])

        self.mae_train_cv = np.mean(mae_train_tmp)
        self.mae_test_cv = np.mean(mae_test_tmp)

        self.std_train_cv = np.std(mae_train_tmp)
        self.std_test_cv = np.std(mae_test_tmp)

        return (self.mae_train_cv, self.mae_test_cv)        


    def get_summary(self):
        """
        Get summary of results:
        
        args:
            summary (dict)
        """

        self.summary = {'results': self.results,
                        'n_fold': self.n_fold,
                        'mae_train_cv': self.mae_train_cv,
                        'mae_test_cv': self.mae_test_cv,
                        'std_train_cv': self.std_train_cv,
                        'std_test_cv': self.std_test_cv}

        return self.summary
            

class Loop_QML_KRR:

    """
    Class for doing QML over different training set sizes.
    """

    def __init__(self, qml_krr):
        """
        Create Loop QML object.

        args:
            qml_krr : QML_KRR oject
        """

        self.qml_krr = qml_krr

        self.train_sizes = None
        
        self.mae_train = None
        self.mae_test = None

        return


    def add_params(self, **params):
        """
        Add Loop QML parameters.
        default:
        train_sizes = [1000, 2000, 4000, 8000, 16000, 32000]
        
        args:
            train_sizes: train_sizes
        """

        default_train_sizes = np.logspace(0, 5, num=6, base=2, endpoint=True) * 1000

        if 'train_sizes' in params:
            self.train_sizes = params['train_sizes']
        else:
            self.train_sizes = default_train_sizes

        return


    def set_train_sizes(self, train_sizes):
        """
        Set the training set sizes:

        args:
            train_sizes (1d ndarray) : training set sizes
        """

        self.train_sizes = train_sizes

        return 

    
    def run(self):
        """
        Iterate over different training set sizes:

        returns:
            results (dict)
        """

        results = {}

        for i, n_train in enumerate(self.train_sizes):

            # split into training and test set

            train_ind, test_ind = self.qml_krr.get_train_test_indices(n_train)
            Y_train, Y_test = self.qml_krr.split_labels()
            kernel_train, kernel_pred = self.qml_krr.split_kernel()

            # actual regression starts here
            
            alpha = self.qml_krr.fit()
            Y_pred_train, Y_pred_test = self.qml_krr.predict()
            mae_train, mae_test = self.qml_krr.evaluate()

            # save the results
            
            results[n_train] = copy(self.qml_krr.get_summary())

        self.results = results
    
        return self.results


    def evaluate(self):
        """
        Evaluate results from QML.

        return:
            mae_train (1d ndarray): mean absolute training error for training set sizes
            mae_test (1d ndarray): mean absolute test error for test set sizes
        """

        mae_test = []
        mae_train = []
        
        for n_train in self.train_sizes:

            mae_train.append(self.results[n_train]['mae_train'])
            mae_test.append(self.results[n_train]['mae_test'])

        self.mae_test = mae_test
        self.mae_train = mae_train

        return (self.mae_train, self.mae_test)


    def get_summary(self):
        """
        Get summary of QML results:

        returns:
            summary (dict)
        """

        self.summary = {'results': self.results,
                        'train_sizes': self.train_sizes,
                        'mae_train': self.mae_train,
                        'mae_test': self.mae_test}

        return self.summary


class CV_Loop_QML_KRR:

    """
    Class for cross-validated QML Loop
    """

    def __init__(self, loop_qml_krr):
        """
        Create CV LOOP QML KRR object.

        args: 
            loop_qml_krr : Loop_QML_KRR object
        """

        self.loop_qml_krr = loop_qml_krr

        self.n_iter = None

        self.mae_train_av = None
        self.mae_test_av = None
        self.std_err_train = None
        self.std_err_test = None

        return

    
    def add_params(self, **params):
        """
        Add paramters for CVed QML Loop.
        default values:
        n_iter : 20
        seed_iter : 666

        args:
            ** params: parameters
        """

        if 'n_iter' in params:
            self.n_iter = params['n_iter']
        else:
            self.n_iter = 20
        if 'seed_iter' in params:
            self.seed_iter = params['seed_iter']
        else:
            self.seed_iter = 666

        return


    def set_n_iter(self, n_iter):
        """
        Set number of iterations:

        args: 
            n_iter : number of repetations of the Loop
        """

        self.n_iter = n_iter

        return


    def set_seed_iter(self, seed_iter):
        """
        Set the seed used for choosing a different seed for the training and test set
        indices RNG.

        args:
            seed_iter : seed
        """

        self.seed_iter = seed_iter

        return


    def update_seed(self):
        """
        Update the seed for every new iteration.

        returns:
            new_seed : new seed
        """

        old_seed = self.loop_qml_krr.qml_krr.seed
        new_seed = int(self.seed_iter / old_seed)
        self.loop_qml_krr.qml_krr.set_seed(new_seed)

        return new_seed


    def run(self):
        """
        Perform repeatitions of QML iterations.

        returns:
            results (dict)
        """

        results = []

        for i in range(self.n_iter):

            loop_results = self.loop_qml_krr.run()
        
            loop_mae_train, loop_mae_test = self.loop_qml_krr.evaluate()

            results.append(copy(self.loop_qml_krr.get_summary()))

            new_seed = self.update_seed()

        self.results = results

        return self.results


    def evaluate(self):
        """
        Evaluate QML results:
        
        returns:
            mae_train_av : average mean absolute error on training set
            mea_test_av : average mean absolute error on test set
        """

        mae_train_tmp = []
        mae_test_tmp = []

        for i in range(self.n_iter):

            mae_train_tmp.append(self.results[i]['mae_train'])
            mae_test_tmp.append(self.results[i]['mae_test'])

        self.mae_train_av = np.mean(np.asarray(mae_train_tmp), axis=0)
        self.mae_test_av = np.mean(np.asarray(mae_test_tmp), axis=0)

        self.std_err_train = np.std(np.asarray(mae_train_tmp), axis=0)
        self.std_err_test = np.std(np.asarray(mae_test_tmp), axis=0)

        return (self.mae_train_av, self.mae_test_av)


    def get_summary(self):
        """
        Get summary of QML results:
        
        returns: 
            summary (dict)
        """
        
        self.summary = {'results': self.results,
                        'mae_train_av': self.mae_train_av,
                        'mae_test_av': self.mae_test_av,
                        'std_err_train': self.std_err_train,
                        'std_err_test': self.std_err_test,
                        'n_iter': self.n_iter}

        return self.summary
