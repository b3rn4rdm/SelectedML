import numpy as np
import pickle
import sys
import os

from qml.kernels import gaussian_kernel_symmetric, laplacian_kernel_symmetric

from qml_krr import *

from qml_classes_qm9_cv import group_by_label, get_train_test_splits

from uuid import uuid4
from hashlib import md5


def load_data(datadir, dataname):
    data = np.load(f'{datadir}{dataname}')
    X = data['X']
    G = data['G']
    H = data['H']
    L = data['L']
    labels = data['labels']
    U = data['U']
    R = data['R']
    return (X, G, H, L, labels, U, R)


def run(X, Y, indices, test_ind, qml_krr, n_train, seed_qml, seed_iter, n_iter):

    results = {}

    new_seed = seed_qml

    for i in range(n_iter):

        # print(f'\tStarting iteration {i}...')

        # print(f'\t\tSplitting training from test set...')
        np.random.seed(new_seed)
        train_ind = np.random.choice(indices, n_train, replace=False)

        qml_krr.add_train_test_indices(train_ind, test_ind)
        qml_krr.split_input()
        qml_krr.split_labels()
        X_train = X[train_ind]
        Y_train = Y[train_ind]
        kernel_train = qml_krr.compute_kernel_train()
        kernel_pred = qml_krr.compute_kernel_pred()

        # print(f'\t\t...fitting alphas...')
        alpha = qml_krr.fit()

        # print(f'\t\t...preditions...')
        Y_pred_train, Y_pred_test = qml_krr.predict()
        mae_train, mae_test = qml_krr.evaluate()
        res_summary = qml_krr.get_summary()
        results[i] = copy(res_summary)

        print(f'\tIteration {i} : MAE = {mae_test} eV')

        # print('\t\t...resetting the seed...')
        new_seed = int(new_seed / seed_iter)
        # print('\t\t..done!')

        # print('\t...done!')

    mae_train_av, mae_test_av, std_err_train, std_err_test = evaluate(results, n_iter)

    return get_summary(results, mae_train_av, mae_test_av, std_err_train, std_err_test,
                       n_iter, seed_iter)


def evaluate(results, n_iter):

    mae_train_iter = []
    mae_test_iter = []
    
    for i in range(n_iter):

        mae_train_iter.append(results[i]['mae_train'])
        mae_test_iter.append(results[i]['mae_test'])

    mae_train_av = np.mean(mae_train_iter)
    mae_test_av = np.mean(mae_test_iter)

    std_err_train = np.std(mae_train_iter)
    std_err_test = np.std(mae_test_iter)

    return (mae_train_av, mae_test_av, std_err_train, std_err_test)


def get_summary(results, mae_train_av, mae_test_av, std_err_train, std_err_test,
                n_iter, seed_iter):

    res_summary = {'results': results,
                   'mae_train_av': mae_train_av, 'mae_test_av': mae_test_av,
                   'std_err_train': std_err_train, 'std_err_test': std_err_test,
                   'n_iter': n_iter, 'seed_iter': seed_iter}

    return res_summary


def run_splits(X, Y, indices, test_ind_subsets, qml_krr, n_train, seed_qml, seed_iter, n_iter,
               subsets_test):
    """
    same as above but runs over multiple test set splits
    """

    results_iter = {}

    new_seed = seed_qml

    for i in range(n_iter):

        # print(f'\tStarting iteration {i}...')

        # print(f'\t\tSplitting training from test set...')
        np.random.seed(new_seed)
        train_ind = np.random.choice(indices, n_train, replace=False)

        X_train = X[train_ind]
        Y_train = Y[train_ind]

        if qml_krr.ktype == 'laplacian':
            kernel_train = laplacian_kernel_symmetric(X_train, qml_krr.sigma)
        elif qml_krr.ktype == 'gaussian':
            kernel_train = gaussian_kernel_symmetric(X_train, qml_krr.sigma)

        alpha = cho_solve(kernel_train + qml_krr.lam * np.eye(n_train), Y_train)

        results_iter[i] = {}
        
        for j, test_ind in enumerate(test_ind_subsets):

            # split again, the train indices are the same but the test indices have changed
            X_test = X[test_ind]
            Y_test = Y[test_ind]
            if qml_krr.ktype == 'laplacian':
                kernel_pred = laplacian_kernel(X_test, X_train, qml_krr.sigma)
            elif qml_krr.ktype == 'gaussian':
                kernel_pred = gaussian_kernel(X_test, X_train, qml_krr.sigma)

            Y_pred_train = np.dot(kernel_train, alpha)
            Y_pred_test = np.dot(kernel_pred, alpha)
            
            mse_train, mse_test, mae_train, mae_test = evaluate_splits(Y_train, Y_test,
                                                                       Y_pred_train, Y_pred_test)
            res_summary = get_summary_splits(mse_train, mse_test, mae_train, mae_test,
                                             Y_train, Y_test, Y_pred_train, Y_pred_test,
                                             train_ind, test_ind, n_train, new_seed)
            
            results_iter[i][j] = copy(res_summary)

            print(f'\tIteration {i}, subset {j} : MAE = {mae_test} eV')

        # print('\t\t...resetting the seed...')
        new_seed = int(new_seed / seed_iter)
        # print('\t\t..done!')

        # print('\t...done!')

    results_re = rearrange_results(results_iter, n_iter, subsets_test)
    results_sub = {}
    for sub_test, res_sub in results_re.items():
        mae_train_av, mae_test_av, std_err_train, std_err_test = evaluate_iter(res_sub, n_iter)
        results_sub[sub_test] = get_summary_iter(res_sub, mae_train_av, mae_test_av,
                                                 std_err_train, std_err_test, n_iter, seed_iter)

    return results_sub


def rearrange_results(results_iter, n_iter, subsets_test):

    results_new = {}
    for sub in subsets_test:
        results_new[sub] = {}
        for i in range(n_iter):
            results_new[sub][i] = results_iter[i][sub]

    return results_new    


def evaluate_splits(Y_train, Y_test, Y_pred_train, Y_pred_test):

    mse_train = np.mean(Y_train - Y_pred_train)
    mse_test = np.mean(Y_test - Y_pred_test)

    mae_train = np.mean(np.abs(Y_train - Y_pred_train))
    mae_test = np.mean(np.abs(Y_test - Y_pred_test))

    return (mse_train, mse_test, mae_train, mae_test)


def evaluate_iter(results, n_iter):

    mae_train_iter = []
    mae_test_iter = []
    for i in range(n_iter):
        mae_train_iter.append(results[i]['mae_train'])
        mae_test_iter.append(results[i]['mae_test'])

    mae_train_av = np.mean(mae_train_iter)
    mae_test_av = np.mean(mae_test_iter)

    std_err_train = np.std(mae_train_iter)
    std_err_test = np.std(mae_test_iter)

    return (mae_train_av, mae_test_av, std_err_train, std_err_test)


def get_summary_splits(mse_train, mse_test, mae_train, mae_test, Y_train, Y_test,
                       Y_pred_train, Y_pred_test, train_ind, test_ind, n_train, seed):

    res_summary = {'mse_train': mse_train, 'mse_test': mse_test,
                   'mae_train': mae_train, 'mae_test': mae_test,
                   'Y_train': Y_train, 'Y_test': Y_test,
                   'Y_pred_train': Y_pred_train, 'Y_pred_test': Y_pred_test,
                   'train_ind': train_ind, 'test_ind': test_ind,
                   'n_train': n_train, 'seed': seed}

    return res_summary


def get_summary_iter(results, mae_train_av, mae_test_av, std_err_train, std_err_test,
                     n_iter, seed_iter):

    results_sum = {'results': results,
                   'mae_train_av': mae_train_av, 'mae_test_av': mae_test_av,
                   'std_err_train': std_err_train, 'std_err_test': std_err_test,
                   'n_iter': n_iter, 'seed_iter': seed_iter}
    
    return results_sum


def main(datadir, dataname, outputdir, outputname, prop, seed_test, seed_qml, seed_iter,
         ktype, n_iter, lam_exp, n_train, comment=''):

    seed_test = int(seed_test)
    seed_qml = int(seed_qml)
    seed_iter = int(seed_iter)
    n_iter = int(n_iter)
    # sigma = float(sigma)
    n_train = int(n_train)
    lam_exp = int(lam_exp)

    # load the dataset
    X, G, H, L, labels, U, R = load_data(datadir, dataname)

    if prop == 'G':
        Y = G
    elif prop == 'H':
        Y = H
    elif prop == 'L':
        Y = L

    X_subsets, u_labels = group_by_label(X, labels)
    Y_subsets, u_labels = group_by_label(Y, labels)
    # G_subsets, u_labels = group_by_label(G, labels)
    # H_subsets, u_labels = group_by_label(H, labels)
    # L_subsets, u_labels = group_by_label(L, labels)

    max_train = 64000
    max_train_sizes = np.asarray([32000, 32000, 16000])
    class_sizes = np.asarray([sub.shape[0] for sub in X_subsets])

    train_ind, test_ind, train_ind_subsets, test_ind_subsets = get_train_test_splits(class_sizes, labels, max_train, max_train_sizes, seed_test)

    lam = 10 ** (lam_exp)
    params = {'lam': lam, 'ktype': ktype}
    qml_krr = QML_KRR()
    qml_krr.add_params( ** params)

    if n_train <= 16000:
        subsets = [-1, 0, 1, 2]
    elif n_train == 32000:
        subsets = [-1, 0, 1]
    elif n_train == 64000:
        subsets = [-1]

    subsets_test = [0, 1, 2]

    print(f'n_train = {n_train} | subsets = {subsets}')

    sigmas = {-1: 256.0, 0: 256.0, 1: 256.0, 2: 256.0}

    for sub in subsets:

        print(f'Calculating for subset {sub}.')
    
        if (sub == -1):
            X_samples = X
            Y_samples = Y
            qml_krr.set_sigma(sigmas[sub])
            qml_krr.add_input(X_samples)
            qml_krr.add_labels(Y_samples)
            results = {}
            # for sub_test in subsets_test:
            #     results[sub_test] = run(X_samples, Y_samples, train_ind, test_ind[sub_test], qml_krr,
            #                             n_train, seed_qml, seed_iter, n_iter)
            results = run_splits(X_samples, Y_samples, train_ind, test_ind, qml_krr,
                                 n_train, seed_qml, seed_iter, n_iter, subsets_test)
        else:
            X_samples = X_subsets[sub]
            Y_samples = Y_subsets[sub]
            qml_krr.set_sigma(sigmas[sub])
            qml_krr.add_input(X_samples)
            qml_krr.add_labels(Y_samples)
            results = run(X_samples, Y_samples, train_ind_subsets[sub], test_ind_subsets[sub],
                          qml_krr, n_train, seed_qml, seed_iter, n_iter)
    
        # add ktype to results
        results['ktype'] = ktype

        results['subset'] = sub
        results['prop'] = prop

        # add comment to results
        # results['comment'] = comment

        # save resutls
        uid = uuid4().hex
        results['uuid'] = uid

        sig = sigmas[sub]
        results['sigma'] = sig
        results['seed_test'] = seed_test
        results['seed_qml'] = seed_qml
        results['seed_iter'] = seed_iter
        results['type'] = 'loop'
        results['dataset'] = 'qm9'

        hash_str = f'qm9 loop {prop} {sub} {R} {ktype} {sig} {lam_exp} {n_train}'
        res_hash = md5(bytes(hash_str, 'utf-8')).hexdigest()
        results['hash'] = res_hash

        with open(f'{outputdir}{outputname}_{res_hash}.pkl', 'wb') as outf:
            pickle.dump(results, outf)

    return


if __name__ == '__main__':

    main( * sys.argv[1:])


