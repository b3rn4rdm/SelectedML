##############################################33
#
# Extract results from QML results to plot learning curves
#
###############################################33

import numpy as np
import os
import sys
import pickle
from copy import copy
from hashlib import md5


def get_prop_list(resdata):

    prop_list = []

    for res in resdata:
        if res['prop'] not in prop_list:
            prop_list.append(res['prop'])

    return prop_list


def get_subset_list(resdata, prop):

    subset_list = []
    for res in resdata:
        if res['prop'] == prop:
            if res['subset'] not in subset_list:
                subset_list.append(res['subset'])

    return subset_list


def get_averaged_mae(res, n_iter, train_sizes):
    mae_all = []
    for i in range(n_iter):
        mae_n_train = []
        for n_train in train_sizes:
            mae_n_train.append(res[i][n_train]['mae_test'])
        mae_all.append(np.asarray(mae_n_train))
    mae_all = np.asarray(mae_all)
    return {'mae_av': mae_all.mean(axis=0), 'std_err': mae_all.std(axis=0)}


def get_res_learning_curves(res, subset, n_iter, train_sizes):
    
    if subset == -1:
        res_sum = {}
        for sub_test in [0, 1, 2]:
            res_sum[sub_test] = get_averaged_mae(res[sub_test], n_iter, train_sizes)
    else:
        res_sum = get_averaged_mae(res, n_iter, train_sizes)

    return res_sum


def get_res_scatter_data(res, subset, iter_ind, train_sizes):

    if subset == -1:
        res_sum = {}
        for sub_test in [0, 1, 2]:
            y_ref = {}
            y_pred = {}
            test_ind = {}
            train_ind = {}
            for n_train in train_sizes:
                y_ref[n_train] = res[sub_test][iter_ind][n_train]['Y_test']
                y_pred[n_train] = res[sub_test][iter_ind][n_train]['Y_pred_test']
                train_ind[n_train] = res[sub_test][iter_ind][n_train]['train_ind']
                test_ind[n_train] = res[sub_test][iter_ind][n_train]['test_ind']
            res_sum[sub_test] = {'Y_ref': y_ref, 'Y_pred': y_pred,
                                 'train_ind': train_ind, 'test_ind': test_ind}
    else:
        y_ref = {}
        y_pred = {}
        train_ind = {}
        test_ind = {}
        for n_train in train_sizes:
            y_ref[n_train] = res[iter_ind][n_train]['Y_test']
            y_pred[n_train] = res[iter_ind][n_train]['Y_pred_test']
            train_ind[n_train] = res[iter_ind][n_train]['train_ind']
            test_ind[n_train] = res[iter_ind][n_train]['test_ind']
        res_sum = {'Y_ref': y_ref, 'Y_pred': y_pred,
                   'train_ind': train_ind, 'test_ind': test_ind}
        
    return res_sum           


# correction necessary for the classification to be coherent with qm9
def swap_classes(res_summary, prop_list):

    for prop in prop_list:
        res_1_old = res_summary[prop][1]
        res_summary[prop][1] = res_summary[prop][2]
        res_summary[prop][2] = res_1_old

    return res_summary


def save_learning_curves(res_sum, prop, sub, R, ktype, sig, lam_exp, res_hash,
                         train_sizes, outputdir, outputfile):

    res_sum['prop'] = prop
    res_sum['sub'] = sub
    res_sum['R'] = R
    res_sum['ktype'] = ktype
    res_sum['sig'] = sig
    res_sum['lam_exp'] = lam_exp
    res_sum['hash'] = res_hash
    res_sum['train_sizes'] = train_sizes

    filename = f'{outputdir}{outputfile}_lc_{res_hash}.pkl' 
    with open(filename, 'wb') as outf:
        pickle.dump(res_sum, outf)

    print(f'Wrote results to {filename}...')

    return 0


def get_results_for_scatter_plot(resdata, prop_list, subset_list, iter_ind):
    
    res_scatter = {}
    for prop in prop_list:
        data_prop = {}
        for subset in subset_list:
            data_tmp = {}
            for res in resdata:
                if (res['prop'] == prop) & (res['subset'] == subset):
                    train_sizes = res['results'][iter_ind]['train_sizes']
                    for n_train in train_sizes:
                        res_sub = res['results'][iter_ind]['results'][n_train]
                        data_tmp[n_train] = {'Y_train': res_sub['Y_train'],
                                             'Y_test': res_sub['Y_test'],
                                             'Y_pred_train': res_sub['Y_pred_train'],
                                             'Y_pred_test': res_sub['Y_pred_test']}
                    data_tmp['uuid'] = res['uuid']
            data_prop[subset] = copy(data_tmp)
        res_scatter[prop] = data_prop
        # res_scatter['uuid'] = res['uuid']
    return res_scatter


def save_scatter_data(res_sum, prop, sub, R, ktype, sig, lam_exp, res_hash,
                      outputdir, outputfile):

    res_sum['prop'] = prop
    res_sum['sub'] = sub
    res_sum['R'] = R
    res_sum['ktype'] = ktype
    res_sum['sig'] = sig
    res_sum['lam_exp'] = lam_exp
    res_sum['hash'] = res_hash
    
    filename = f'{outputdir}{outputfile}_scatter_{res_hash}.pkl'
    with open(filename, 'wb') as outf:
        pickle.dump(res_sum, outf)
    print(f'Wrote results to {filename}...')

    return 0


def main(inputdir, inputname, outputdir, outputname, iter_ind):

    iter_ind = int(iter_ind)

    # prop_list = ['G_ZINDO', 'G_PBE0', 'G_GW',
    #              'H_ZINDO', 'H_PBE0', 'H_GW',
    #              'L_ZINDO', 'L_PBE0', 'L_GW']

    prop_list = ['G_ZINDO']
    
    # for delta learning 
    # prop_pairs = [('G_ZINDO', 'G_GW'), ('H_ZINDO', 'H_GW'), ('L_ZINDO', 'L_GW'),
    #               ('G_ZINDO', 'G_PBE0'), ('H_ZINDO', 'H_PBE0'), ('L_ZINDO', 'L_PBE0'),
    #               ('G_PBE0', 'G_GW'),  ('H_PBE0', 'H_GW'),  ('L_PBE0', 'L_GW')]

    prop_pairs = [('G_ZINDO', 'G_GW')]
    
    subsets = [-1, 0, 1, 2]
    R = 'cm' # or 'bob' or 'slatm'
    ktype = 'laplacian'
    sigmas = {-1: 128.0, 0: 512.0, 1: 256.0, 2: 256.0}
    lam_exp = -12
    n_iter = 10

    train_sizes_dict = {-1: [100, 200, 400, 800, 1600, 3200, 6400],
                        0: [100, 200, 400, 800, 1600],
                        1: [100, 200, 400, 800, 1600],
                        2: [100, 200, 400, 800]}

    n_train_dict = {-1: 6400, 0: 1600, 1: 1600, 2:800}

    # uncomment whatever line you need
    # comment whatever line you don't need
    
    for prop in prop_list:
    # for pair in prop_pairs:
        print(f'Fetching results for property {prop}.')
        # prop_base = pair[0]
        # prop_target = pair[1]
        # prop = pair
        # print(f'Fetching results for property pair {prop_base} {prop_target}.')
        for sub in subsets:
            print(f'Fetching results for subset {sub}.')
            sig = sigmas[sub]
            hash_str = f'qm7b loop {prop} {sub} {R} {ktype} {sig} {lam_exp}'
            # hash_str = f'qm7b loop delta {prop_base} {prop_target} {sub} {R} {ktype} {sig} {lam_exp}'
            res_hash = md5(bytes(hash_str, 'utf-8')).hexdigest()
            with open(f'{inputdir}{inputname}_{res_hash}.pkl', 'rb') as inf:
                res = pickle.load(inf)

            res_sum_lc = get_res_learning_curves(res, sub, n_iter, train_sizes_dict[sub])
            save_learning_curves(res_sum_lc, prop, sub, R, ktype, sig, lam_exp, res_hash,
                                 train_sizes_dict[sub], outputdir, outputname)

            res_sum_scatter = get_res_scatter_data(res, sub, iter_ind, train_sizes_dict[sub])
            save_scatter_data(res_sum_scatter, prop, sub, R, ktype, sig, lam_exp, res_hash,
                              outputdir, outputname)
            
    print('Done!')
    
    return


if __name__ == "__main__":

    main( * sys.argv[1:])

