conda create -n drmvsnet python=3.6
source activate drmvsnet
conda install -c daleydeng gcc-5 -y
conda install pytorch=1.1.0 cuda90 -c pytorch -y
pip install opencv-python
pip install plyfile

