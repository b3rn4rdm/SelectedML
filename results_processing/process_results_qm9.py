import numpy as np
import os
import sys
import pickle
import copy
import tarfile


from process_results_qm7b import save_learning_curves, save_scatter_data

from hashlib import md5


def get_averaged_mae(res, n_iter):

    mae_list = []
    for i in range(n_iter):
        mae_list.append(res[i]['mae_test'])
    mae_test_av = np.mean(mae_list)
    std_err = np.std(mae_list)

    return (mae_test_av, std_err)


def get_res_learning_curves(res_list, subset, n_iter, train_sizes):

    if subset == -1:
        res_sum = {}
        for sub_test in [0, 1, 2]:
            mae_tmp = []
            std_err_tmp = []
            for n_train in train_sizes:
                mae_av, std_err = get_averaged_mae(res_list[n_train][sub_test]['results'], n_iter)
                mae_tmp.append(mae_av)
                std_err_tmp.append(std_err_tmp)
            res_sum[sub_test] = {'mae_av': mae_tmp, 'std_err': std_err_tmp}
    else:
        mae_tmp = []
        std_err_tmp = []
        for n_train in train_sizes:
            mae_av, std_err = get_averaged_mae(res_list[n_train]['results'], n_iter)
            mae_tmp.append(mae_av)
            std_err_tmp.append(std_err_tmp)
        res_sum = {'mae_av': mae_tmp, 'std_err': std_err_tmp}
                
    return res_sum


def get_res_scatter_data(res_list, subset, iter_ind, train_sizes):

    if subset == -1:
        res_sum = {}
        for sub_test in [0, 1, 2]:
            y_ref = {}
            y_pred = {}
            test_ind = {}
            train_ind = {}
            for n_train in train_sizes:
                y_ref[n_train] = res_list[n_train][sub_test]['results'][iter_ind]['Y_test']
                y_pred[n_train] = res_list[n_train][sub_test]['results'][iter_ind]['Y_pred_test']
                train_ind[n_train] = res_list[n_train][sub_test]['results'][iter_ind]['train_ind']
                test_ind[n_train] = res_list[n_train][sub_test]['results'][iter_ind]['test_ind']
            res_sum[sub_test] = {'Y_ref': y_ref, 'Y_pred': y_pred,
                                 'train_ind': train_ind, 'test_ind': test_ind}
    else:
        y_ref = {}
        y_pred = {}
        test_ind = {}
        train_ind = {}
        for n_train in train_sizes:
            y_ref[n_train] = res_list[n_train]['results'][iter_ind]['Y_test']
            y_pred[n_train] = res_list[n_train]['results'][iter_ind]['Y_pred_test']
            train_ind[n_train] = res_list[n_train]['results'][iter_ind]['train_ind']
            test_ind[n_train] = res_list[n_train]['results'][iter_ind]['test_ind']
        res_sum = {'Y_ref': y_ref, 'Y_pred': y_pred,
                   'train_ind': train_ind, 'test_ind': test_ind}

    return res_sum
                

def main(inputdir, inputname, outputdir, outputname, iter_index):

    iter_index = int(iter_index)

    subsets = [-1, 0, 1, 2]

    # prop_list = ['G', 'H', 'L']
    # prop_list = ['G', 'H']
    prop_list = ['G']
    R = 'cm' # or 'slatm' or 'bob'
    ktype = 'laplacian'
    sigmas = {-1: 256.0, 0: 256.0, 1: 256.0, 2: 256.0}
    lam_exp = -12
    n_iter = 10

    # train_sizes_dict = {-1: [1000, 2000, 4000, 8000, 16000, 32000, 64000],
    #                     -0: [1000, 2000, 4000, 8000, 16000, 32000],
    #                     1: [1000, 2000, 4000, 8000, 16000, 32000],
    #                     2: [1000, 2000, 4000, 8000, 16000]}

    # dummy, just for testing
    train_sizes_dict = {-1: [4000], 0: [4000], 1: [4000], 2: [4000]}    

    for prop in prop_list:
        
        print('Fetching results for property {prop}.')
        
        for sub in subsets:
            
            print(f'\tFetching results for subset {sub}.')
            
            sig = sigmas[sub]
            res_tmp = {}
            for n_train in train_sizes_dict[sub]:
                hash_str = f'qm9 loop {prop} {sub} {R} {ktype} {sig} {lam_exp} {n_train}'
                print(hash_str)
                res_hash = md5(bytes(hash_str, 'utf-8')).hexdigest()
                with open(f'{inputdir}{inputname}_{res_hash}.pkl', 'rb') as inf:
                    res_tmp[n_train] = pickle.load(inf)
                    
            res_sum_lc = get_res_learning_curves(res_tmp, sub, n_iter, train_sizes_dict[sub])
            save_learning_curves(res_sum_lc, prop, sub, R, ktype, sig, lam_exp, res_hash,
                                 train_sizes_dict[sub], outputdir, outputname)
            
            res_sum_scatter = get_res_scatter_data(res_tmp, sub, iter_index, train_sizes_dict[sub])
            save_scatter_data(res_sum_scatter, prop, sub, R, ktype, sig, lam_exp, res_hash,
                              outputdir, outputname)


    return


if __name__ == '__main__':

    main( * sys.argv[1:])
