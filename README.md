# jaxns-cosmology

```bash
DOWNLOAD_DIR=$HOME
INSTALL_DIR=$HOME
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $DOWNLOAD_DIR/miniconda.sh
bash $DOWNLOAD_DIR/miniconda.sh -b -p $INSTALL_DIR/miniconda3
. $INSTALL_DIR/miniconda3/etc/profile.d/conda.sh
echo ". $INSTALL_DIR/miniconda3/etc/profile.d/conda.sh" >> $HOME/.bashrc
hash -r 
conda config --set auto_activate_base false --set always_yes yes
conda update -q conda
conda info -a
```

#### Create new Conda environment


```bash
conda create -n jaxns_cosmology_py python=3.10
conda activate jaxns_cosmology_py
pip install -r requirements.txt
```

#### Compile and install pymultinest and pypolychord


```bash
sudo apt install liblapack-dev libblas-dev openmpi-*

git clone https://github.com/JohannesBuchner/MultiNest
cd MultiNest/build
cmake ..
make

git clone https://github.com/PolyChord/PolyChordLite.git
cd PolyChordLite
make MPI=0
pip install .

# Test the installation
python -c 'import pymultinest'
python -c 'import pypolychord'
```