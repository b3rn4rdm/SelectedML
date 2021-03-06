##############################3
#
# Let a ML program classify qm7b data
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

from qml_classes_qm7b_cv import group_by_label

from qml_classify_qm9 import classify


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
        G_ZINDO, H_ZINDO, L_ZINDO = Y[:, 0], Y[:, 1], Y[:, 2]
        G_PBE0, H_PBE0, L_PBE0 = Y[:, 3], Y[:, 4], Y[:, 5]
        G_GW, H_GW, L_GW = Y[:, 6], Y[:, 7], Y[:, 8]        
        rep_data_new = {'X': X, 'G_ZINDO': G_ZINDO, 'H_ZINDO': H_ZINDO, 'L_ZINDO': L_ZINDO,
                        'G_PBE0': G_PBE0, 'H_PBE0': H_PBE0, 'L_PBE0': L_PBE0,
                        'G_GW': G_GW, 'H_GW': H_GW, 'L_GW': L_GW, 
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
    G_ZINDO = rep_data['G_ZINDO']
    G_PBE0 = rep_data['G_PBE0']
    G_GW = rep_data['G_GW']
    H_ZINDO = rep_data['H_ZINDO']
    H_PBE0 = rep_data['H_PBE0']
    H_GW = rep_data['H_GW']
    L_ZINDO = rep_data['L_ZINDO']
    L_PBE0 = rep_data['L_PBE0']
    L_GW = rep_data['L_GW']
    Y_ZINDO = np.hstack((G_ZINDO[:, np.newaxis], H_ZINDO[:, np.newaxis], L_ZINDO[:, np.newaxis]))
    Y_PBE0 = np.hstack((G_PBE0[:, np.newaxis], H_PBE0[:, np.newaxis], L_PBE0[:, np.newaxis]))
    Y_GW = np.hstack((G_GW[:, np.newaxis], H_GW[:, np.newaxis], L_GW[:, np.newaxis]))
    Y = np.hstack((Y_ZINDO, Y_PBE0, Y_GW))
    labels =  rep_data[f'labels']
    data_uuid = rep_data['U']

    train_sizes = np.asarray(np.logspace(0, 6,  num=7, endpoint=True, base=2) * 100).astype(int)

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

