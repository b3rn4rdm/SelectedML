import numpy as np
import pickle
import sys
import os

from qml.kernels import gaussian_kernel_symmetric, laplacian_kernel_symmetric

from qml_krr import *

from qml_classes_qm7b_cv import group_by_label, get_train_test_splits
from qml_classes_qm7b_cv_loop import get_loop_summary, get_iter_summary

from uuid import uuid4
from hashlib import md5


def load_data(datadir, dataname, prop_base, prop_target):

    data = np.load(f'{datadir}{dataname}')

    X = data['X']
    Y_base = data[prop_base]
    Y = data[prop_target]
    U = data['U']
    labels = data['labels']
    R = data['R']

    return (X, Y_base, Y, U, labels, R)


def run_delta_qml(X, Y_base, Y, indices, test_ind, delta_qml_krr, train_sizes, seed_qml, seed_iter, n_iter):
    """
    args:
        X : input representation
        Y_base : 
        Y : target values
        indices : indices from which to choose the training samples
        delta_qml_krr : delta_qml_krr object
        train_sizes : list with training set sizes
        seed_qml : initial seed for the RNG
        seed_iter : extra seed to generate new RNGs
        n_iter : number of iterations
    """

    # split away the test set (these indices will always remain the same anyway)
    X_test = X[test_ind]
    Y_base_test = Y_base[test_ind]
    Y_test = Y[test_ind]

    np.random.seed(seed_qml)
    new_seed = seed_qml

    results = {}
    
    # loop over the training set sizes
    for i in range(n_iter):

        print(f'\tresults for iteration {i}:')

        res_n_train = {}

        for n_train in train_sizes:

            # randomly choose n_train indices from the training indices
            train_ind = np.random.choice(indices, n_train, replace=False)
    
            # split the kernel into training and test kernel
            delta_qml_krr.add_train_test_indices(train_ind, test_ind)
            delta_qml_krr.split_input()
            delta_qml_krr.split_labels()
            X_train = X[train_ind]
            Y_base_train = Y_base[train_ind]
            Y_train = Y[train_ind]
            kernel_train, kernel_pred = delta_qml_krr.split_kernel()
            
            # compute the alpha coefficients
            alpha = delta_qml_krr.fit()
            
            # compute the predictions
            Y_pred_train, Y_pred_test = delta_qml_krr.predict()

            # evaluate the results
            mae_train, mae_test = delta_qml_krr.evaluate()

            print(f'\t\tfor train size {n_train}: MAE = {mae_test} eV')

            # save the results
            res_n_train[n_train] = copy(delta_qml_krr.get_summary())

        results[i] = res_n_train
        mae_train_loop, mae_test_loop = get_loop_summary(res_n_train, train_sizes)
        results[i]['mae_train'] = mae_train_loop
        results[i]['mae_test'] = mae_test_loop

        # reinitialize a new RNG
        new_seed = int(new_seed / seed_iter)
        np.random.seed(new_seed)

    mae_train_av, mae_test_av, std_err_train, std_err_test = get_iter_summary(results, n_iter)
    results['mae_train_av'] = mae_train_av
    results['mae_test_av'] = mae_test_av
    results['std_err_train'] = std_err_train
    results['std_err_test'] = std_err_test
    
    return results    


def main(datadir, dataname, outputdir, outputname, prop_base, prop_target, seed_test, seed_qml,
         seed_iter, ktype, n_iter, lam_exp, comment=''):
    """
    args: 
        datadir : directory of dataset file
        dataname : name of dataset file
        outputdir : directory for output data
        outputname : name of output data
        prop_base : key for baseline values 
        porp_target : key for target values
        seed_test
        seed_qml : seed for random training and test set split
        seed_iter
        ktype : kernel to be used
        n_iter
        lam_exp
    """

    seed_test= int(seed_test)
    seed_qml = int(seed_qml)
    seed_iter = int(seed_iter)
    n_iter = int(n_iter)
    lam_exp = int(lam_exp)
        
    # load the dataset
    X, Y_base, Y, U, labels, R = load_data(datadir, dataname, prop_base, prop_target)

    X_subsets, u_labels = group_by_label(X, labels)
    Y_base_subsets, u_labels = group_by_label(Y_base, labels)
    Y_subsets, u_labels = group_by_label(Y, labels)

    class_sizes = np.asarray([sub.shape[0] for sub in X_subsets])
    max_train_sizes = np.asarray([1600, 1600, 800])
    max_train = 6400

    train_ind, test_ind, train_ind_subsets, test_ind_subsets = get_train_test_splits(class_sizes, labels, max_train, max_train_sizes, seed_qml)

    lam = 10 ** (lam_exp)
    params = {'lam': lam, 'ktype': ktype}
    delta_qml_krr = Delta_QML_KRR()
    delta_qml_krr.add_params( ** params)

    subsets = [-1, 0, 1, 2]
    sigmas = {-1: 128.0, 0: 512.0, 1: 256.0, 2: 256.0}

    # generate kernels

    for sub in subsets:

        if sub == -1:
            if ktype == 'gaussian':
                kernel = gaussian_kernel_symmetric(X, sigmas[sub])
            elif ktype == 'laplacian':
                kernel = laplacian_kernel_symmetric(X, sigmas[sub])
            delta_qml_krr.set_sigma(sigmas[sub])
            delta_qml_krr.add_input(X)
            delta_qml_krr.add_labels(Y)
            delta_qml_krr.add_kernel(kernel)
            delta_qml_krr.add_base_labels(Y_base)
            train_sizes = [100, 200, 400, 800, 1600, 3200, 6400]
            res_tmp = {}
            for sub_test in [0, 1, 2]:
                res_tmp[sub_test] = run_delta_qml(X, Y_base, Y, train_ind, test_ind[sub_test],
                                                  delta_qml_krr, train_sizes, seed_qml, seed_iter,
                                                  n_iter)
            results = res_tmp
        else:
            if ktype == 'gaussian':
                kernel = gaussian_kernel_symmetric(X_subsets[sub], sigmas[sub])
            elif ktype == 'laplacian':
                kernel = laplacian_kernel_symmetric(X_subsets[sub], sigmas[sub])
            delta_qml_krr.set_sigma(sigmas[sub])
            delta_qml_krr.add_input(X_subsets[sub])
            delta_qml_krr.add_labels(Y_subsets[sub])
            delta_qml_krr.add_kernel(kernel)
            delta_qml_krr.add_base_labels(Y_base_subsets[sub])
            if sub == 2:
                train_sizes = [100, 200, 400, 800]
            else:
                train_sizes = [100, 200, 400, 800, 1600]
            results = run_delta_qml(X_subsets[sub], Y_base_subsets[sub], Y_subsets[sub],
                                    train_ind_subsets[sub], test_ind_subsets[sub],
                                    delta_qml_krr, train_sizes, seed_qml, seed_iter, n_iter)

        results['subset'] = sub

        # add comment to results
        # results['comment'] = comment
        results['prop_base'] = prop_base
        results['prop_target'] = prop_target
        results['R'] = R
        results['seed_test'] = seed_test
        results['type'] = 'loop delta'
        results['dataset'] = 'qm7b'

        sig = sigmas[sub]

        hash_str = f'qm7b loop delta {prop_base} {prop_target} {sub} {R} {ktype} {sig} {lam_exp}'
        res_hash = md5(bytes(hash_str, 'utf-8')).hexdigest()
        results['hash'] = res_hash

        # save resutls
        uid = uuid4().hex
        results['uuid'] = uid
        
        with open(f'{outputdir}{outputname}_{res_hash}.pkl', 'wb') as outf:
            pickle.dump(results, outf)
    
    return


if __name__ == '__main__':

    main( * sys.argv[1:])
