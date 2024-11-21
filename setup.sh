set -e

ENV_NAME=cespar

conda create -n $ENV_NAME -y -c opensim-org -c conda-forge opensim==4.4 python=3.7
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

conda install -y libgfortran-ng==11.2.0 libgfortran5==11.2.0 libffi==3.3 simbody==3.7
conda install -y -c conda-forge lapack git
pip install git+https://github.com/stanfordnmbl/osim-rl@610b95cf0c4484f1acecd31187736b0113dcfb73
pip install -r requirements.txt
conda install -y mpi4py==3.0.3

echo
echo conda activate $ENV_NAME
echo 
