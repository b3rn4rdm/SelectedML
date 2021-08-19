import numpy as np
import pickle
import sys
import os

from qml.kernels import gaussian_kernel_symmetric, laplacian_kernel_symmetric

from qml_krr import *

from uuid import uuid4
from hashlib import md5


def load_data(datadir, dataname, prop):
    """
    load representations and target values

    args: 
        datadir : directory of dataset file
        dataname : name of dataset file
        prop : key for lumo values ('L_ZINDO', 'L_PBE0', 'L_GW')
        labels : label values

    return: 
        X : representation vector
        Y : corresponding lumo values
        U : uuid of dataset
    """
    
    data = np.load(f'{datadir}{dataname}')
    X = data['X']
    Y = data[prop]
    U = data['U']
    labels = data['labels']
    R = data['R']
    
    return (X, Y, U, labels, R)


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


def get_rand_indices(n_samples, n_train, seed_qml):

    indices = np.arange(n_samples).astype(int)
    np.random.seed(seed_qml)
    np.random.shuffle(indices)

    return indices[:n_train]


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



def main(datadir, dataname, outputdir, outputname, prop, seed_test, seed_train, seed_cv,
         ktype, n_fold, lam_exp, comment=''):
    """
    args: 
        datadir : directory of dataset file
        dataname : name of dataset file
        outputdir : directory for output data
        outputname : name of output data
        prop : key for lumo values ('L_ZINDO', 'L_PBE0', 'L_GW')
        seed_test : seed for random training and test set split
        seed_train 
        seed_cv : 
        ktype : kernel to be used   
        n_fold : n_fold
        lam_exp
    """

    seed_test = int(seed_test)
    seed_train = int(seed_train)
    seed_cv = int(seed_cv)
    n_fold = int(n_fold)
    lam_exp = int(lam_exp)
    
    # load the dataset
    X, Y, U, labels, R = load_data(datadir, dataname, prop)

    X_subsets, u_labels = group_by_label(X, labels)
    Y_subsets, u_labels = group_by_label(Y, labels)

    max_train = 6400
    max_train_sizes = np.asarray([1600, 1600, 800])
    class_sizes = np.asarray([sub.shape[0] for sub in X_subsets])

    train_ind, test_ind, train_ind_subsets, test_ind_subsets = get_train_test_splits(class_sizes, labels, max_train, max_train_sizes, seed_test)
    # print(train_ind.shape)

    # generate kernels
    sigmas = np.logspace(0, 12, num=13, base=2, endpoint=True)

    # set up parameters for QML
    lam = 10 ** (lam_exp)

    # define params
    qml_params = {'lam': lam, 'ktype': ktype, 'seed': seed_cv}
    cv_params = {'n_fold': n_fold}

    subsets = [-1, 0, 1, 2]

    for sub in subsets:

        print(f'Running CV for subset {sub}:')

        results = {}

        for i, sig in enumerate(sigmas):
        
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
    
            # create qml object
            qml_krr = QML_KRR()
            qml_krr.add_params(**qml_params)
            qml_krr.add_kernel(kernel)
            qml_krr.add_labels(Y_samples)
            
            cv_qml_krr = CV_QML_KRR(qml_krr)
            cv_qml_krr.add_params(**cv_params)
    
            results_cv = cv_qml_krr.run()
            mae_train, mae_test = cv_qml_krr.evaluate()

            print(f'\tsigma = {sig} : MAE : {mae_test} eV')
            
            results[sig] = cv_qml_krr.get_summary()
        
        results['subset'] = sub
        results['seed_test'] = seed_test
        results['seed_train'] = seed_train
        results['prop'] = prop
        # results_summary['label'] = label

        # add comment to results
        results['comment'] = comment

        # save resutls
        uid = uuid4().hex
        results['uuid'] = uid
        hash_str = f'qm7b cv {prop} {sub} {R} {ktype} {lam_exp}'
        res_hash = md5(bytes(hash_str, 'utf-8')).hexdigest()
        results['R'] = R
        results['type'] = 'cv'
        results['dataset'] = 'qm7b'
        results['hash'] = res_hash
        
        with open(f'{outputdir}{outputname}_{res_hash}.pkl', 'wb') as outf:
            pickle.dump(results, outf)
    
    return


if __name__ == '__main__':

    main( * sys.argv[1:])
