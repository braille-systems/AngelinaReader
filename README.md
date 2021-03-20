# Angelina braille reader

[![Actions Status](https://github.com/braille-systems/AngelinaReader/workflows/Python%20CI/badge.svg)](https://github.com/braille-systems/topological-sorting/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code Coverage](https://codecov.io/gh/braille-systems/AngelinaReader/branch/develop/graph/badge.svg)](https://codecov.io/gh/braille-systems/topological-sorting)
[![CodeFactor](https://www.codefactor.io/repository/github/braille-systems/angelinareader/badge/develop)](https://www.codefactor.io/repository/github/braille-systems/angelinareader/overview/develop)

A modification of [IlyaOvodov/AngelinaReader](https://github.com/IlyaOvodov/AngelinaReader)

# Requirements
- Git with Git LFS
- Python 3.6-3.8

# Setup for development
## Installation
1. Run from the command line (Tested on Ubuntu and Msys2/MinGW):
    ```
    git clone --recursive https://github.com/braille-systems/AngelinaReader.git
    cd AngelinaReader
    git clone --recursive https://github.com/braille-systems/brl_ocr.git
    python -m pip install --upgrade pip
    ```
    Substitute `python` with your system's Python3 command, which may be `python`, `py` or `python3`.
1. Add a new virtual environment: 
    
    ```
      sudo apt install python3-venv
      python -m venv env
      source env/bin/activate
    ```
    
    (on Linux) or 
    
    ```
      python -m pip install virtualenv
      python -m venv env
      .\env\Scripts\activate
    ```
    
    (on Windows)
    
    **Note:** to work on this project, you need to re-activate the environment every time you restart the machine or open a new terminal session.
    To verify that you've successfully activated the environment, run `which python` (or `where python` on Windows).
    You should see `<...>\AngelinaReader\env\Scripts\python.exe` on Windows (and something alike on other platforms) at the top of the list.
1. Install the dependencies:
    ```
   python -m pip install --upgrade pip
   python -m pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
   python -m pip install -r requirements.txt
   python -m pip install -r model/requirements.txt
   ```

## Running pre-trained inference

To ensure that you've done everything correctly, you may launch a pre-trained model:

``` 
wget -O weights/model.t7 http://angelina-reader.ovdv.ru/retina_chars_eced60.clr.008
python run_local.py brl_ocr/data/unlabeled/golubina/IMG_2503.JPG
```

If you're running Linux and getting errors like "Found no NVIDIA driver on your system", try to install NVIDIA CUDA Toolkit:
```
sudo apt install nvidia-cuda-toolkit
```
(I must admit that I still did not succeed in running training on Linux, so maybe some additional steps will be required)

## Training

Execute the file:
    ```
    python model/train.py
    ```
   If you're getting CUDA errors, you may change `device="cuda:0"` to `device="cpu"` in `model/params.py`
   
   This will run for a while (~20 hours with NVIDIA GeForce GTX 1660 GPU, more than 24 hours with 1080Ti).
   Results will appear under `NN_results` folder.
   
## Labels

See [brl_ocr/README.md#working-with-labels](https://github.com/braille-systems/brl_ocr#working-with-labels).

It's OK to install labelme in the same environment as other dependencies in this project.
