# conda create 3ds-vla python=3.8
conda create -n 3ds-vla python=3.8 -y
conda init
conda activate 3ds-vla
python -V
# export COPPELIASIM_ROOT=./CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 # follow the repo from peract to download CoppeliaSim
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
# export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
pip install -r requirements.txt
cd 3ds-vla/YARR
pip install -r requirements.txt
pip install -e . 
cd ../RLBench 
pip install -r requirements.txt
pip install -e . 
cd ../PyRep
pip install -r requirements.txt
pip install -e . 
pip uninstall opencv-python #need opencv headless for simumation
pip uninstall opencv-python-headless
pip install opencv-python-headless==4.12.0.88
pip install open3d==0.18.0
pip install -e git+https://github.com/facebookresearch/pytorch3d.git@055ab3a2e3e611dff66fa82f632e62a315f3b5e7#egg=pytorch3d
pip install scikit-image==0.21.0
