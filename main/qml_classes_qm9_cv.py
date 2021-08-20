import numpy as np
import pickle
import sys
import os

from qml.kernels import gaussian_kernel, laplacian_kernel
from qml.kernels import gaussian_kernel_symmetric, laplacian_kernel_symmetric
from sklearn.model_selection import train_test_split

from qml_krr import *

from uuid import uuid4
from hashlib import md5


def load_data(inputdir, inputname, prop):
    data = np.load(f'{inputdir}{inputname}')
    X = data['X']
    Y = data[prop]
    labels = data['labels']
    U = data['U']
    R = data['R']
    return (X, Y, labels, U, R)


def group_by_label(a, labels):
    a_subsets = []
    u_labels = np.unique(labels)
    for ul in u_labels:
        a_tmp = []
        for ai, l in zip(a, labels):
            if l == ul:
                a_tmp.append(ai)
        a_subsets.append(np.asarray(a_tmp))
    return (a_subsets, u_labels)


def choose_samples(X, G, nsamples, seed_select):
    np.random.seed(seed_select)
    indices_tot = np.arange(len(X)).astype(int)
    np.random.shuffle(indices_tot)
    indices_samples = indices_tot[:nsamples]
    X_samples = X[indices_samples]
    G_samples = G[indices_samples]
    return (X_samples, G_samples)


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

    # retrace corresponding indices of the original dataset
    unique_labels = np.sort(np.unique(labels))
    label_indices = []
    for ul in unique_labels:
        label_indices.append(np.where(labels == ul)[0])

    original_test_indices = []
    original_train_indices = []
    for lindices, test_ind, train_ind in zip(label_indices, test_ind_subsets, train_ind_subsets):
        original_test_indices.append(lindices[test_ind])
        original_train_indices.append(lindices[train_ind])

    all_test_indices = np.concatenate(original_test_indices)
    all_train_indices = np.concatenate(original_train_indices)

    # print(all_test_indices.shape, all_train_indices.shape)

    assert np.intersect1d(all_train_indices, all_test_indices).shape[0] == 0, 'error in train test splitting!'

    # create indices for the training sets
    return (all_train_indices, original_test_indices, train_ind_subsets, test_ind_subsets)


def do_qml(X_train, X_test, Y_train, Y_test, kernel, sigma, seed_cv, lam, n_folds, data_uuid,
           **kwargs):

    if kernel == 'gaussian':
        kernel_train = gaussian_kernel_symmetric(X_train, sigma)
        kernel_pred = gaussian_kernel(X_test, X_train, sigma)
    elif kernel == 'laplacian':
        kernel_train = laplacian_kernel_symmetric(X_train, sigma)
        kernel_pred = laplacian_kernel(X_test, X_train, sigma)

    print(f'CV for sigma = {sigma}: ')

    res_cv = qml_cross_validate(kernel_train, Y_train, sigma, data_uuid, n_folds, lam, seed_cv)

    mae_list = np.asarray([res_cv[k]['test']['mae'] for k in range(n_folds)])
    std_list = np.asarray([res_cv[k]['test']['std'] for k in range(n_folds)])
    med_list = np.asarray([res_cv[k]['test']['med'] for k in range(n_folds)])

    # compute regression coefficients
    if ('method' in kwargs) & ('cond' in kwargs):
        alpha = qml_fit(kernel_train, Y_train, lam, method, cond)
    else:
        alpha = qml_fit(kernel_train, Y_train, lam)
        
    # make predictions
    Y_pred_train = qml_predict(kernel_train, alpha)
    Y_pred_test = qml_predict(kernel_pred, alpha)
    
    # evaluate results
    mae_train, std_train, med_train = qml_evaluate(Y_pred_train, Y_train)
    mae_test, std_test, med_test = qml_evaluate(Y_pred_test, Y_test)

    print(f'Error on test set with sigma = {sigma}', mae_test)

    results = {'train': {'mae': mae_train, 'std': std_train, 'med': med_train,
                         'ref': Y_train, 'pred': Y_pred_train},
               'test': {'mae': mae_test, 'std': std_test, 'med': med_test,
                        'ref': Y_test, 'pred': Y_pred_test},
               'cv': {'mae': mae_list, 'std': std_list, 'seed': seed_cv}}
    
    results['uuid'] = uuid4().hex

    print('\n')

    return results


def get_rand_indices(n_samples, n_train, seed_qml):

    indices = np.arange(n_samples).astype(int)
    np.random.seed(seed_qml)
    np.random.shuffle(indices)

    return indices[:n_train]


def main(inputdir, inputname, outputdir, outputname, prop, seed_test, seed_train, seed_cv,
         ktype, n_fold, lam_exp):
    """
    args : 
        inputdir
        inputname
        outputdir
        outputname
        prop
        seed_test
        seed_train
        seed_cv
        ktype
        n_fold
        lam_exp
    """
    
    seed_test = int(seed_test)
    seed_train = int(seed_train)
    seed_cv = int(seed_cv)
    n_fold = int(n_fold)
    lam_exp = int(lam_exp)
    
    # load the dataset
    X, Y, labels, U, R = load_data(inputdir, inputname, prop)

    X_subsets, u_labels = group_by_label(X, labels)
    Y_subsets, u_labels = group_by_label(Y, labels)

    max_train = 6400
    max_train_sizes = np.asarray([3200, 3200, 1600])
    class_sizes = np.asarray([sub.shape[0] for sub in X_subsets])

    train_ind, test_ind, train_ind_subsets, test_ind_subsets = get_train_test_splits(class_sizes, labels, max_train, max_train_sizes, seed_test)

    # parameters ofr qml
    sigmas = np.logspace(7, 11, num=5, base=2, endpoint=True)
    lam = 10 ** (lam_exp)

    qml_params = {'lam': lam, 'ktype': ktype, 'seed': seed_cv}
    cv_params = {'n_fold': n_fold}

    subsets = [-1, 0, 1, 2]

    for sub in subsets:

        print(f'Running CV for subset {sub}: ')

        results = {}

        for i, sig in enumerate(sigmas):

            # print(f'\t Compute kernels for sigma = {sig}')

            if sub == -1:
                np.random.seed(seed_train)
                rand_indices = np.random.choice(train_ind, max_train, replace=False)
                X_samples = X[rand_indices]
                Y_samples = Y[rand_indices]
                if ktype == 'gaussian':
                    kernel = gaussian_kernel_symmetric(X_samples, sig)
                elif ktype == 'laplacian':
                    kernel = laplacian_kernel_symmetric(X_samples, sig)
            else:
                np.random.seed(seed_train)
                rand_indices = np.random.choice(train_ind_subsets[sub], max_train_sizes[sub],
                                                replace=False)
                X_samples = X_subsets[sub][rand_indices]
                Y_samples = Y_subsets[sub][rand_indices]
                if ktype == 'gaussian':
                    kernel = gaussian_kernel_symmetric(X_samples, sig)
                elif ktype == 'laplacian':
                    kernel = laplacian_kernel_symmetric(X_samples, sig)
            
            qml_params['sigma'] = sig

            # create QML object
            qml_krr = QML_KRR()
            qml_krr.add_params( ** qml_params)
            qml_krr.add_input(X_samples)
            qml_krr.add_labels(Y_samples)
            qml_krr.add_kernel(kernel)

            cv_qml_krr = CV_QML_KRR(qml_krr)
            cv_qml_krr.add_params( ** cv_params)
            
            result_cv = cv_qml_krr.run()
            mae_train_cv, mae_test_cv = cv_qml_krr.evaluate()

            print(f'\tsigma = {sig} : MAE = {mae_test_cv} eV')
            
            results[sig] = cv_qml_krr.get_summary()

        results['subset'] = sub
        results['prop'] = prop
        results['seed_test'] = seed_test
        results['seed_train'] = seed_train

        uid = uuid4().hex
        results['uuid'] = uid
        hash_str = f'qm9 cv {prop} {sub} {R} {ktype} {lam_exp}'
        res_hash = md5(bytes(hash_str, 'utf-8')).hexdigest()
        results['R'] = R
        results['type'] = 'cv'
        results['dataset'] = 'qm9'
        results['hash'] = res_hash
        
        # save resutls
        with open(f'{outputdir}{outputname}_{res_hash}.pkl', 'wb') as outf:
            pickle.dump(results, outf)

    return


if __name__ == '__main__':

    main( * sys.argv[1:])
