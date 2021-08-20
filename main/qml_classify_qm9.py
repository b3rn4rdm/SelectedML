##############################3
#
# Let a ML program classify qm9 data
#
###################################



import numpy as np
import os
import sys
import pickle

from uuid import uuid4
from datetime import datetime

from sklearn import tree, svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.gaussian_process.kernels import RBF

from qml_classes_qm9_cv import group_by_label


def classify(classifier, X, Y, labels, train_sizes, seed):
    """
    Classify the molecules into subsets according to the given labels:
    
    args: 
        classifier (string) : classifier to use ('svm', 'tree')
        X : representations
        Y : target values
        labels : labels
        train_sizes : training set sizes
        seed : seed for the random choice of training and test samples

    return: 
        results: results as a dict
    """

    results = {'seed': seed, 'classifier': classifier, 'train_sizes': train_sizes}

    for i, ts in enumerate(train_sizes):


        if classifier == 'svm':

            length_scales = np.logspace(0, 8, num=9, endpoint=True, base=2)
            C = np.logspace(0, 4, num=5, endpoint=True, base=2)
            rbf_kernels = [RBF(l) for l in length_scales]
            param_grid = {'C': C, 'kernel': rbf_kernels}
            est = GridSearchCV(svm.SVC(gamma=1), param_grid=param_grid)

        elif classifier == 'tree':

            est = tree.DecisionTreeClassifier()

        X_train, X_test, Y_train, Y_test, labels_train, labels_test = train_test_split(X, Y, labels, train_size=ts, random_state=seed)
        est.fit(X_train, labels_train)
        labels_pred = est.predict(X_test)
        s = est.score(X_test, labels_test)

        print(f'Score for training set size {ts}: {s}')

        results[ts] = {'score': s, 'X_test': X_test, 'Y_test': Y_test,
                       'labels_test': labels_test, 'labels_pred': labels_pred}

    return results


def save_results(resdir, outputdir, outputname, results):
    """
    save the results with predicted labels:

    args: 
        resdir : directory for the results file
        outputdir : directory for the new representation files
        outputname : filename
        results : dict with results

    returns:
        None
    """

    train_sizes = results['train_sizes']
    seed = results['seed']
    classifier = results['classifier']
    results_classification = {'seed': seed, 'train_sizes': train_sizes, 'classifier': classifier}

    if os.path.exists(outputdir) == False:
        os.makedirs(outputdir)


    for i, ts in enumerate(train_sizes):
        results_classification[ts] = {'score': results[ts]['score'],
                                      'labels_pred': results[ts]['labels_pred']}
        X = results[ts]['X_test']
        labels = results[ts]['labels_pred']
        labels_ref = results[ts]['labels_test']
        Y = results[ts]['Y_test']
        G = Y[:, 0]
        H = Y[:, 1]
        L = Y[:, 2]
        rep_data_new = {'X': X, 'G': G, 'H': H, 'L': L,
                        f'labels': labels,
                        f'labels_ref': labels_ref}
        for key in ['U', 'D', 'R']:
            rep_data_new[key] = results[key]
        filename = f'{outputdir}{outputname}_%06i.npz'%ts
        np.savez(filename, **rep_data_new)


    with open(f'{resdir}{outputname}.pkl', 'wb') as outf:
        pickle.dump(results_classification, outf)
 
    return
        
            

def main(inputdir, inputname, resdir, outputdir, outputname, classifier, seed):

    seed = int(seed)
    
    rep_data = np.load(f'{inputdir}{inputname}')
    X = rep_data['X']
    G = rep_data['G']
    H = rep_data['H']
    L = rep_data['L']
    Y = np.hstack((G[:, np.newaxis], H[:, np.newaxis], L[:, np.newaxis]))
    labels =  rep_data[f'labels']
    data_uuid = rep_data['U']

    train_sizes = np.logspace(0, 7, num=8, endpoint=True, base=2).astype(int) * 1000

    print('Starting classification ...')
    results = classify(classifier, X, Y, labels, train_sizes, seed)
    print('...done!')
    
    for key in ['U', 'R']:
        results[key] = rep_data[key]
    dt = datetime.now()
    timestr = dt.strftime('%Y%m%d%H%M%S')
    results['D'] = timestr

    print('Saving results...')
    save_results(resdir, outputdir, outputname, results)
    print('...done!')

    return 


if __name__ == '__main__':

    main( * sys.argv[1:])
