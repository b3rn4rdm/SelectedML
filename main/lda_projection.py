   ################################3
#
# Perform LDA on dataset and project onto main components
# with highest variance between classes and save them
#
##################################


import numpy as np
import os
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def main(inputdir, inputname, outputdir, outputfile, scale='False', qm9='True'):
    """
    args: 
        inputdir (string) : 
        inputname (string) : 
        outputdir (string) : 
        outputfile (string) : 
        scaled=False (boolean) :
        qm9=True (boolean)
    """

    data = np.load(f'{inputdir}{inputname}')

    X = data['X']
    labels = data['labels']

    C = data['C']
    Z = data['Z']

    if qm9 == 'True':
        G = data['G']
        H = data['H']
        L = data['L']
    else:
        G_ZINDO = data['G_ZINDO']
        H_ZINDO = data['H_ZINDO']
        L_ZINDO = data['L_ZINDO']
        G_PBE0 = data['G_PBE0']
        H_PBE0 = data['H_PBE0']
        L_PBE0 = data['L_PBE0']
        G_GW = data['G_GW']
        H_GW = data['H_GW']
        L_GW = data['L_GW']

    lda = LDA(n_components=2, store_covariance=True)
    if scale == 'True':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_trans = lda.fit_transform(X_scaled, labels)
    else:
        X_trans = lda.fit_transform(X, labels)

    cov = lda.covariance_
    means = lda.means_
    evr = lda.explained_variance_ratio_
    uid = data['U']

    with open(f'{outputdir}{outputfile}', 'wb') as outf:
        if qm9 == 'True':
            np.savez(outf, X_trans=X_trans, labels=labels, G=G, H=H, L=L, U=uid,
                     Z=Z, Co=C, 
                     C=cov, M=means, E=evr)
        else:
            np.savez(outf, X_trans=X_trans, labels=labels, Z=Z, Co=C, 
                     G_ZINDO=G_ZINDO, H_ZINDO=H_ZINDO, L_ZINDO=L_ZINDO,
                     G_PBE0=G_PBE0, H_PBE0=H_PBE0, L_PBE0=L_PBE0,
                     G_GW=G_GW, H_GW=H_GW, L_GW=L_GW, U=uid,
                     C=cov, M=means, E=evr)
            
    return


if __name__ == '__main__':

    main( * sys.argv[1:])
