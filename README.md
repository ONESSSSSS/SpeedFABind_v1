<h1 align="center">
Code repository for the paper 《SpeedFABind:Accelerating Protein-Ligand Pocket Prediction and Docking Based on DGL via Three-Tier Optimization》
</h1>




## Setup Environment
This is an example of how to set up a working conda environment to run the code. In this example, we have cuda version==11.3, torch==1.12.0, and rdkit==2021.03.4. To make sure the pyg packages are installed correctly, we directly install them from whl.

**As the trained model checkpoint is included in the HuggingFace repository with git-lfs, you need to install git-lfs to pull the data correctly.**

```shell
sudo apt-get install git-lfs # run this if you have not installed git-lfs
git lfs install
git clone https://github.com/QizhiPei/FABind.git --recursive
conda create --name fabind python=3.8
conda activate fabind
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_cluster-1.6.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_sparse-0.6.15%2Bpt112cu113-cp38-cp38-linux_x86_64.whl 
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_spline_conv-1.2.1%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/pyg_lib-0.2.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install torch-geometric==2.4.0
pip install torchdrug==0.1.2 torchmetrics==0.10.2 tqdm mlcrate pyarrow accelerate Bio lmdb fair-esm tensorboard
pip install fair-esm
pip install rdkit-pypi==2021.03.4
conda install -c conda-forge openbabel # install openbabel to save .mol2 file and .sdf file at the same time
```

## Data
The PDBbind 2020 dataset can be download from http://www.pdbbind.org.cn. We then follow the same data processing as [TankBind](https://github.com/luwei0917/TankBind/blob/main/examples/construction_PDBbind_training_and_test_dataset.ipynb).

We also provided processed dataset on [zenodo](https://zenodo.org/records/11352521).
If you want to train FABind from scratch, or reproduce the FABind results, you can:
1. download dataset from [zenodo](https://zenodo.org/records/11352521)
2. unzip the `zip` file and place it into `data_path` such that `data_path=pdbbind2020`

### Generate the ESM2 embeddings for the proteins
Before training or evaluation, you need to first generate the ESM2 embeddings for the proteins based on the preprocessed data above.
```shell
data_path=pdbbind2020

python fabind/tools/generate_esm2_t33.py ${data_path}
```
Then the ESM2 embedings will be saved at `${data_path}/dataset/processed/esm2_t33_650M_UR50D.lmdb`.


## Evaluation
```shell
data_path=pdbbind2020
ckpt_path=ckpt/best_model.bin

python fabind/test_fabind.py \
    --batch_size 4 \
    --data-path $data_path \
    --resultFolder ./results \
    --exp-name test_exp \
    --ckpt $ckpt_path \
    --local-eval
```

## Inference on Custom Complexes
Here are the scripts available for inference with smiles and according pdb files.

The following script iteratively runs:
- Given smiles in `index_csv`, preprocess molecules with `num_threads` multiprocessing and save each processed molecule to `{save_pt_dir}/mol`.
- Given protein pdb files in `pdb_file_dir`, preprocess protein information and save it to `{save_pt_dir}/processed_protein.pt`.
- Load model checkpoint in `ckpt_path`, save the predicted molecule conformation in `output_dir`. Another csv file in `output_dir` indicates the smiles and according filename.

```shell
index_csv=../inference_examples/example.csv
pdb_file_dir=../inference_examples/pdb_files
num_threads=1
save_pt_dir=../inference_examples/temp_files
save_mols_dir=${save_pt_dir}/mol
ckpt_path=../ckpt/best_model.bin
output_dir=../inference_examples/inference_output

cd fabind

echo "======  preprocess molecules  ======"
python inference_preprocess_mol_confs.py --index_csv ${index_csv} --save_mols_dir ${save_mols_dir} --num_threads ${num_threads}

echo "======  preprocess proteins  ======"
python inference_preprocess_protein.py --pdb_file_dir ${pdb_file_dir} --save_pt_dir ${save_pt_dir}

echo "======  inference begins  ======"
python fabind_inference.py \
    --ckpt ${ckpt_path} \
    --batch_size 4 \
    --seed 128 \
    --test-gumbel-soft \
    --redocking \
    --post-optim \
    --write-mol-to-file \
    --sdf-output-path-post-optim ${output_dir} \
    --index-csv ${index_csv} \
    --preprocess-dir ${save_pt_dir} \
    --sdf-to-mol2
```
