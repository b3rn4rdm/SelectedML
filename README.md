# MolClass_HOMO_LUMO

Molecular Classification Improves Data-efficiency of Quantum Machine Learning Models of HOMO-LUMO gaps

For technical reasons, there are separate scripts for both datasets.

## Frequency Analysis

   python frequency_analysis.py inputdir/ inputname outputdir/ outputname dataset=['qm9', 'qm7b'] smiles=['smiles', 'xyz']


## Generate Data for QML

For QM7b:

    python generate_qml_data_qm7b.py input_dir/ inputname freq_analysis_name outputdir/ outputname rep=['cm', 'bob', 'slatm']
   
For QM9:

   python generate_qml_data.py input_dir/ inputname freq_analysis_name outputdir/ outputname rep=['cm', 'bob', 'slatm']


## Run Cross-Validation

For QM7b:

    python qml_classes_qm7b_cv.py inputdir/ inputname outputdir/ outputname prop seed_test seed_train seed_cv ktype n_fold lam_exp

For QM9:

    python qml_classes_qm9_cv.py inputdir/ inputname outputdir/ outputname prop seed_test seed_train seed_cv ktype n_fold lam_exp


## Generate Learning Curves

For QM7b:

    python qml_classes_qm7b_cv_loop.py inputdir/ inputname outputdir/ outputname prop seed_test seed_qml seed_iter ktype n_iter lam_exp

For $\Delta$-ML:

    python qml_delta_classes_qm7b_cv_loop.py inputdir/ inputname outputdir/ outputname prop_base prop_target seed_test seed_qml seed _iter ktype n_iter lam_exp

For QM9:

    python qml_classes_qm9_cv_loop.py inputdir/ inputname outputdir/ outputname prop seed_test seed_qml seed_iter ktype n_iter lam_exp n_train


## Process Results

For QM7b:

    python process_results_qm7b.py inputdir/ inputname outputdir/ outputname iter_index

For QM9:

    python process_results_qm9.py inputdir/ inputname outputdir/ outputname iter_index