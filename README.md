# Voice recognition | Machine learning
 
> ⚠ Make sure the audio file is mono (single channel) and in `.wav` format. 

## Prepare python virtual environment

### Via VS Code

0. Open VS Code in the root of the repository.
1. Install the `ms-python.python` extension. 
2. Use `Ctrl`+`Shift`+`P` and find `Python: Create Environment...`.
3. Select `Venv`.
4. Select the newest version of python executable on your machine.
5. Select `requirements.txt`.
6. Click `OK`.

VS Code will create a virtual environment in `.venv` and download all of the required packages automatically.

### Via PyCharm

0. Open the PyCharm in the root of the repository.
1. You will be prompted with a `Creating Virtual Environment` dialog box.
2. Select the newest base interpreter.
3. (Optional) You may rename the `venv` to `.venv` if you want to have compatibility between VS Code and PyCharm.
4. Click `OK`.

PyCharm will create a virtual environment in `venv` (or `.venv`) and download all of the required packages automatically.

### Via terminal
Navigate to the root of the repository first, then run the following commands.

On Linux and MacOS:

```bash
python -m  venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows (CMD):

```bash
python -m  venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

On Windows (Powershell):

```bash
python -m  venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run application
Enter the virtual environment first if you are using the terminal (note that PyCharm and VS Code will do that automatically
in their integrated terminals). Then run the application

```bash
python src/main.py
```

## Development

### Using `DapsExplorer`

Before using the DAPS, it's convenient to put the downloaded data into the `data` folder in the root directory of the repository (it's gitignored).

See `notebook/data_analysis/example/example.ipynb` and `src/main.py` for `DapsExplorer` usage examples. 
