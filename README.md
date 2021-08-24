# MolClass_HOMO_LUMO

Code used to generate the results presented in the article "Molecular Classification Improves Data-efficiency of Quantum Machine Learning Models of HOMO-LUMO gaps" (insert arXiv link here).

Organic molecules from QM9 and QM7b are distributed into 3 classes according to some simple rules based on the functional groups. ML within these classes improves the out-of-sample prediction errors of HOMO-LUMO gaps, as well as HOMO and LUMO energies.

Most of the code is organized into scripts. For technical reasons, there are separate scripts for QM9 and QM7b. Below are explanations how the scripts are used.
- The first line provides the syntax and the second one shows an example.
- All directories have to have a "/" at the end.
- Sometimes, the filenames are with extensions, sometimes without. This is because some scripts generate several output files.
- The output files for the ML results contain a hash generated from a string that contains specific information about that result (dataset, representation, some hyperparameters used, ...). I find this easier to use than to put these values in the file name itself, because otherwise the filenames become excessively long concatenations of parameter values. 
- You may need to create new directories to store your data and results.
- Some variables and hyperparameters have to be changed in the `main` function in the scripts, depending on what you need. That's why it's worth taking a look at the main function of each script before you run it.
- The following packages are required:
  - QMLcode:
  - xyz2mol:

In the folder `data_preprocessing/`: 

## Frequency Analysis

Screens the dataset and generates a list with tags referring to functional groups for each molecule. QM9 contains SMILES strings, so they can be directly used. QM7b however doesn't, so 'xyz2mol' is required to get SMILES string from the coordinates.

   python frequency_analysis.py inputdir/ inputname outputdir/ outputname dataset=['qm9', 'qm7b'] smiles=['smiles', 'xyz']
   e.g.:
   python frequency_analysis.py ../data/ qm9.tar.bz2 ../data/ qm9_frequency_analysis.pkl qm9 smiles

## Generate Data for QML

For QM7b:

Generate `.npz` files with the representations, coordinates, nuclear charges and properties of interest together with the class labels.

    python generate_qml_data_qm7b.py input_dir/ inputname freq_analysis_name outputdir/ outputname rep=['cm', 'bob', 'slatm']
    e.g.:
    python generate_qml_data_qm7b.py ../data/ qm7b.txt qm7b_frequency_analysis.pkl ../results/ qm7b_cm.npz cm 
    
   
For QM9:

   python generate_qml_data.py input_dir/ inputname freq_analysis_name outputdir/ outputname rep=['cm', 'bob', 'slatm']
   e.g.:
   python generate_qml_data.py ../data/ qm9.tar.bz2 qm9_frequency_analysis.pkl ../results/ qm9_cm.npz cm


In the folder `../main/`:

## Run Cross-Validation

Runs a grid-search CV within a given training set size, usually the largest one possible for a given class, in order to determine the optimal kernel width $\sigma$. The parameter space is defined in the `main` function. 3 seeds are required for random selection of the test molecules (`seed_test`), training molecules (`seed_train`) and random splitting of the training set into equally sized folds. The test set molecules are actually ignored in the CV.

For QM7b:

    python qml_classes_qm7b_cv.py inputdir/ inputname outputdir/ outputname prop seed_test seed_train seed_cv ktype n_fold lam_exp
    e.g.:
    python qml_classes_qm7b_cv.py ../data/ qm7b_cm.npz ../results/ qm7b_res G_ZINDO 42 987654321 5 laplacian 5 -12

For QM9:

    python qml_classes_qm9_cv.py inputdir/ inputname outputdir/ outputname prop seed_test seed_train seed_cv ktype n_fold lam_exp
    e.g.:
    python qml_classes_qm9_cv.py ../data/ qm9_cm.npz ../results/ qm9_res G 42 987654321 5 laplacian 5 -12 


## Generate Learning Curves

Runs ML over increasing training set sizes in order to produce learning curves. For each class, a training and test set are generated. In addition, a training set is generated with molecules drawn at random without consideration of class labels. For each test set, we produce 2 predictions, one from the model trained on the corresponding class ans a second one from the model that ignores class labels. The kernel width $\sigma$ is specified in the `main` function and can be obtained from the previous CV. The whole procedure is repeated 'n_iter' times, with different trainign set moelcules for each repetition. Again, 3 seeds have to be specified: `seed_test` for chhosing the test set molecules, `seed_qml` for choosing training set molecules and `seed_iter` for making sure that at every repetition different molecules are chosen for training. Note that `seed_iter` needs to be much smaller than `seed_qml`. For each repetation, a new seed is calculated by dividing `seed_qml` by `seed_iter`, thus guaranteeing different training sets for each repretition.

For QM7b:

    python qml_classes_qm7b_cv_loop.py inputdir/ inputname outputdir/ outputname prop seed_test seed_qml seed_iter ktype n_iter lam_exp
    e.g.:
    python qml_classes_qm7b_cv_loop.py ../data/ qm7b.npz ../results/ qm9_res G_ZINDO 42 66 5 laplacian 10 -12
    
For $\Delta$-ML:

    python qml_delta_classes_qm7b_cv_loop.py inputdir/ inputname outputdir/ outputname prop_base prop_target seed_test seed_qml seed_iter ktype n_iter lam_exp
    e.g.:
    python qml_delta_classes_qm7b_cv_loop.py ../data/ qm7b_cm.npz ../results/ qm7b_res G_ZINDO G_GW 42 987654321 5 laplacian 10 -12

For QM9: (Note that the script has to be called for each training set size separately.)

    python qml_classes_qm9_cv_loop.py inputdir/ inputname outputdir/ outputname prop seed_test seed_qml seed_iter ktype n_iter lam_exp n_train
    e.g.:
    python qml_classes_qm9_cv_loop.py ../data/ qm9_cm.npz ../results/ qm9_res G 42 987654321 5 laplacian 10 -12 1000

In the folder `results_processing/`:

## Process Results

These scripts serve to gather and summarize the results that are needed to produde learning curves and scatter plots. The QM7b-script serves to process results from direct and $\Delta$-ML: comment (out) the appropriate lines in the `main` function. `iter_index` specifies which iteration is supposed to be used for the scatter plots.

For QM7b:

    python process_results_qm7b.py inputdir/ inputname outputdir/ outputname iter_index
    e.g.:
    python process_results_qm7b.py ../results/ qm7b_res ../results/ qm7b_res_sum 0

For QM9:

    python process_results_qm9.py inputdir/ inputname outputdir/ outputname iter_index
    e.g.:
    python process_results_qm9.py ../results/ qm9_res ../results/ qm9_res_sum 0