import numpy as np
import pickle
import sys
import os

from qml.kernels import gaussian_kernel_symmetric, laplacian_kernel_symmetric

from qml_krr import *

from qml_classes_qm7b_cv import load_data, group_by_label

from uuid import uuid4
from hashlib import md5


def load_data(datadir, dataname, prop):

    data = np.load(f'{datadir}{dataname}')

    X = data['X']
    Y = data[prop]
    U = data['U']
    labels = data['labels']
    R = data['R']

    return (X, Y, U, labels, R)


def get_train_test_splits(class_sizes, labels, max_train, max_train_sizes, seed_test):
    """
    args: 
        class_sizes : list with the sizes (int) of the classes
        labels : list with labels
        max_train : largest training set size overall
        max_train_sizes : largets training set sizes for the respective classes
        seed_test : seed for the random index generation
    """

    # identify indices of the test splits within the classes
    test_ind_subsets = []
    train_ind_subsets = []
    for cs, mts in zip(class_sizes, max_train_sizes):
        tss = cs - mts
        # print(tss)
        train_ind, test_ind = train_test_split(np.arange(cs).astype(int),
                                               test_size=tss, random_state=seed_test)
        # print(train_ind.shape, test_ind.shape)
        train_ind_subsets.append(train_ind)
        test_ind_subsets.append(test_ind)

    # get additional training indices to get up to max_train
    n_all_train_ind = np.sum([max_train_sizes])
    n_train_ind_left = max_train - n_all_train_ind
    n_test_ind = np.sum([sub.shape[0] for sub in test_ind_subsets])

    new_train_ind_subsets = []
    new_test_ind_subsets = []
    for train_ind, test_ind in zip(train_ind_subsets, test_ind_subsets):
        n_choice = int((test_ind.shape[0] + 1) / n_test_ind * n_train_ind_left)
        train_ind_tmp = np.random.choice(test_ind, n_choice, replace=False)
        new_train_ind_subsets.append(np.concatenate([train_ind, train_ind_tmp]))
        test_ind_tmp = []
        for i in test_ind:
            if i not in train_ind_tmp:
                test_ind_tmp.append(i)
        test_ind_tmp = np.asarray(test_ind_tmp)
        new_test_ind_subsets.append(test_ind_tmp)

        # print(train_ind_tmp.shape, test_ind_tmp.shape)

    # retrace corresponding indices of the original dataset
    unique_labels = np.sort(np.unique(labels))
    label_indices = []
    for ul in unique_labels:
        label_indices.append(np.where(labels == ul)[0])

    original_test_indices = []
    original_train_indices = []
    for lindices, test_ind, train_ind in zip(label_indices, new_test_ind_subsets, new_train_ind_subsets):
        original_test_indices.append(lindices[test_ind])
        original_train_indices.append(lindices[train_ind])

    all_test_indices = np.concatenate(original_test_indices)
    all_train_indices = np.concatenate(original_train_indices)

    # print(all_test_indices.shape, all_train_indices.shape)

    assert np.intersect1d(all_train_indices, all_test_indices).shape[0] == 0, 'error in train test splitting!'

    # create indices for the training sets
    return (all_train_indices, original_test_indices, new_train_ind_subsets, new_test_ind_subsets)


def get_loop_summary(results_loop, train_sizes):

    mae_train = []
    mae_test = []

    for n_train in train_sizes:

        mae_train.append(results_loop[n_train]['mae_train'])
        mae_test.append(results_loop[n_train]['mae_test'])

    return (mae_train, mae_test)


def get_iter_summary(results_iter, n_iter):

    mae_train_tmp = []
    mae_test_tmp = []

    for i in range(n_iter):

        mae_train_tmp.append(results_iter[i]['mae_train'])
        mae_test_tmp.append(results_iter[i]['mae_test'])

    mae_train_av = np.mean(np.asarray(mae_train_tmp), axis=0)
    mae_test_av = np.mean(np.asarray(mae_test_tmp), axis=0)

    std_err_train = np.std(np.asarray(mae_train_tmp), axis=0)
    std_err_test = np.std(np.asarray(mae_test_tmp), axis=0)

    return (mae_train_av, mae_test_av, std_err_train, std_err_test)


def run_qml(X, Y, indices, test_ind, qml_krr, train_sizes, seed_qml, seed_iter, n_iter):
    """
    args:
        X : input representation
        Y : target values
        indices : indices from which to choose the training samples
        qml_krr : qml_krr object
        train_sizes : list with training set sizes
        seed_qml : initial seed for the RNG
        seed_iter : extra seed to generate new RNGs
        n_iter : number of iterations
    """

    # split away the test set (these indices will always remain the same anyway)
    X_test = X[test_ind]
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
            qml_krr.add_train_test_indices(train_ind, test_ind)
            qml_krr.split_input()
            qml_krr.split_labels()
            X_train = X[train_ind]
            Y_train = Y[train_ind]
            kernel_train, kernel_pred = qml_krr.split_kernel()
            
            # compute the alpha coefficients
            alpha = qml_krr.fit()
            
            # compute the predictions
            Y_pred_train, Y_pred_test = qml_krr.predict()

            # evaluate the results
            mae_train, mae_test = qml_krr.evaluate()

            print(f'\t\tfor train size {n_train}: MAE = {mae_test} eV')

            # save the results
            res_n_train[n_train] = copy(qml_krr.get_summary())

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


def main(datadir, dataname, outputdir, outputname, prop, seed_test, seed_qml, seed_iter,
         ktype, n_iter, lam_exp, comment=''):
    """
    args: 
        datadir : directory of dataset file
        dataname : name of dataset file
        outputdir : directory for output data
        outputname : name of output data
        prop : key for lumo values ('L_ZINDO', 'L_PBE0', 'L_GW')
        sigma
        seed_test
        seed_qml : seed for random training and test set split
        seed_iter
        ktype : kernel to be used   
        n_iter : 
        lam_exp : exponent for lambda = 1e ^ (lam_exp)
    """

    # sigma = float(sigma)
    seed_test = int(seed_test)
    seed_qml = int(seed_qml)
    seed_iter = int(seed_iter)
    # subset = int(subset)
    lam_exp = int(lam_exp)
    n_iter = int(n_iter)
    
    # load the dataset
    X, Y, U, labels, R = load_data(datadir, dataname, prop)

    X_subsets, u_labels = group_by_label(X, labels)
    Y_subsets, u_labels = group_by_label(Y, labels)

    class_sizes = np.asarray([sub.shape[0] for sub in X_subsets])
    max_train_sizes = np.asarray([1600, 1600, 800])
    max_train = 6400

    train_ind, test_ind, train_ind_subsets, test_ind_subsets = get_train_test_splits(class_sizes, labels, max_train, max_train_sizes, seed_qml)

    lam = 10 ** (lam_exp)
    params = {'lam': lam, 'ktype': ktype}
    qml_krr = QML_KRR()
    qml_krr.add_params( ** params)

    # loop over the subsets
    subsets = [-1, 0, 1, 2]

    sigmas = {-1: '128.0', 0: '512.0', 1: '256.0', 2: '256.0'}
    for sub in subsets:
        # generate the respective kernel matrix

        print(f'results for subset {sub}')
        
        if sub == -1:
            if ktype == 'laplacian':
                kernel = laplacian_kernel_symmetric(X, sigmas[sub])
            elif ktype == 'gaussian':
                kernel = gaussian_kernel_symmetric(X, sigmas[sub])
            qml_krr.set_sigma(sigmas[sub])
            qml_krr.add_input(X)
            qml_krr.add_labels(Y)
            qml_krr.add_kernel(kernel)
            train_sizes = [100, 200, 400, 800, 1600, 3200, 6400]
            res_tmp = {}
            for sub_test in [0, 1, 2]:
                res_tmp[sub_test] = run_qml(X, Y, train_ind, test_ind[sub_test], qml_krr,
                                            train_sizes, seed_qml, seed_iter, n_iter)
            results = res_tmp
        else:
            if ktype == 'laplacian':
                kernel = laplacian_kernel_symmetric(X_subsets[sub], sigmas[sub])
            elif ktype == 'gaussian':
                kernel = gaussian_kernel_symmetric(X_subsets[sub], sigmas[sub])
            qml_krr.set_sigma(sigmas[sub])
            qml_krr.add_input(X_subsets[sub])
            qml_krr.add_labels(Y_subsets[sub])
            qml_krr.add_kernel(kernel)
            if sub == 2:
                train_sizes = [100, 200, 400, 800]
            else:
                train_sizes = [100, 200, 400, 800, 1600]
            results = run_qml(X_subsets[sub], Y_subsets[sub], train_ind_subsets[sub],
                              test_ind_subsets[sub], qml_krr, train_sizes,
                              seed_qml, seed_iter, n_iter)
        
        # save resutls
        uid = uuid4().hex
        results['uuid'] = uid
        sig = sigmas[sub]
        hash_str = f'qm7b loop {prop} {sub} {R} {ktype} {sig} {lam_exp}'
        res_hash = md5(bytes(hash_str, 'utf-8')).hexdigest()
        
        results['subset'] = sub
        results['R'] = R
        results['seed_test'] = seed_test
        results['type'] = 'loop'
        results['dataset'] = 'qm7b'
        results['hash'] = res_hash
                     
        with open(f'{outputdir}{outputname}_{res_hash}.pkl', 'wb') as outf:
            pickle.dump(results, outf)
    
    return


if __name__ == '__main__':

    main( * sys.argv[1:])
