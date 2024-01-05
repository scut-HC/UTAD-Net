# UTAD-Net

1. Download the BRATS2020 dataset.
2. Slice three-dimensional images into two dimensions and resize to 128 * 128.
3. Randomly partitioning the dataset into training, validation, and testing sets.
4. train techer network  
`
python ./train_w_local_branch.py --phase train
`
5. train studet network  
`
python ./train_distill_global_local.py --phase train
`

6. test   
`
python ./train_distill_global_local.py --phase test
`
