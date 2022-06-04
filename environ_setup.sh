conda create -n ai4code
conda activate ai4code

conda install pip
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pandas numpy matplotlib
pip install transformers
pip install scipy wandb tqdm
pip install scikit-learn

conda install jupyterlab
conda install ipykernel

pip install flake8 pytest
