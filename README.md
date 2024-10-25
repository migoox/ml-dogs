# Voice recognition | Machine learning
 
> ⚠ Make sure the audio file is mono (single channel) and in `.wav` format. 

## Install dependencies

```bash
pip install -r src/requirements.txt
```

### Using virtual environment
Some linux distributions prohibit installing global python packages with pip
to avoid overwriting the python packages shipped by the distro package manager.

For that reason it may be helpful to use virtual python environment.

```bash
python -m  venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

Alternatively you may install `ms-python.python` extension to automate this process with VS Code.

## Run application
```bash
python src/main.py
```

## Using `DapsExplorer`

Before using the DAPS is convenient to put the downloaded data into the `data` folder in the root directory of the repository (it's gitignored).

See `notebook/example/example.ipynb` and `src/main.py` for usage examples. 
