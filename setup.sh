# git clone https://github.com/flavell-lab/pytorch-3dunet.git
# git clone https://github.com/flavell-lab/flv_utils.git
# git clone git@github.com:flavell-lab/euler_gpu.git
conda env create -f environment.yaml
cd euler_gpu
pip install .
cd ../pytorch-3dunet/
pip install -e .