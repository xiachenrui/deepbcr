# DeepBCR Project

If you used this software in your work, please cite:
### DeepBCR: Deep learning framework for cancer-type classification and binding affinity estimation using B cell receptor repertoires
##### Authors: Xihao Hu, X Shirley Liu
##### Address: Department of Data Sciences, Dana-Farber Cancer Institute and Harvard T.H. Chan School of Public Health, Boston, MA, USA

## Software Versions

 * Python 3.6 (https://www.python.org/downloads/release/python-365/)
 * TensorFlow 1.5.0 (https://www.tensorflow.org/install/)
 * Numpy 1.14.3 (https://docs.scipy.org/doc/numpy-1.14.0/reference/)
 * Pandas 0.21.0 (https://pandas.pydata.org/)

## Setup the Environment

Install *conda* for python 2.7 (https://conda.io/docs/user-guide/install/download.html)

Create an empty environment in python 3.6
```
conda create -n py36 python=3.6
```

Load the environment and install required packages
```
source activate py36
conda install -y numpy pandas scikit-learn ipython tensorflow  matplotlib seaborn jupyter
```

## Test the Functions

Test all the deep learning models using a synthetic data
```
cd src/
python deep_bcr.py 
```
