# ProgreSpine

Implementation of MICCAI 2025 conference paper: [ProgreSpine: Inherently Explainable Prototypical Regression for Spine Age Estimation](https://link.springer.com/chapter/10.1007/978-3-032-05185-1_48)

Authors: Roozbeh Bazargani, Saqib Basar, Sam Hashemi & Siavash Khallaghi

This package is a 3D implementation of prototypical regression tested on T2-weighted whole-spine MRI for age estimation task.

## Dependencies

### Install Pyenv and Python 3.12.6
This repository requires the standard distribution of Python 3.12.6 (not Conda). The best way to install 
Python 3.12 is to use [Pyenv](https://github.com/pyenv/pyenv-installer). Refer to pyenv Github page for installation instructions. 
NOTE: Make sure to follow [Set up your shell environment for Pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#set-up-your-shell-environment-for-pyenv) so pyenv is properly loaded 
every time you log in to your machine.

Once you've installed pyenv, you can use it to install Python 3.12 on your machine:

```
pyenv install 3.12.6
```

### Install Poetry
To install Poetry issue this command:

```bash
curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.8.5 python3 -
```

Configure poetry to create its virtual environment in the same directory as the pyproject.toml file.

```bash
poetry config virtualenvs.in-project true
```

**Note**: If you encountered `no distribution available` error when installing
Poetry, run `pyenv global 3.12.6`.


### Install Just task runner
To install Just, run the following script:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --tag 1.43.0 --to /home/ubuntu/.local/bin
```

### Clone the repository
```bash
git clone git@github.com:prenuvo/progrespine
cd progrespine
```

Now you can see a list of tasks that you can run with _just_. On the prompt type:
```
just
```

### Install requirements using Poetry
Run the following script:
```
poetry install
```
Now all dependencies are installed. We can move on to running the code.

## Running the code

Training and testing is managed through the provided [`justfile`](justfile).

- `just train` - trains the model
- `just inference --ckpt <path-to-checkpoint-model>` - runs inference on the test set. The `<path-to-checkpoint-model>` should be like `mlruns/187301657032709345/0dfa0a345f5343a18da6f29b0493e5ae/checkpoints/epoch=48-step=69335.ckpt`

## Files in the package - adapting to a new dataset

Here, we only explain the files in the package and what to change in order to run it on a new data.

- `progrespine/models/protonet.py` - includes model code. You might want to update hyperparameters in order to adapt the model to a new data
- `progrespine/dataset/spine.py` - includes spine torch Dataset and Dataloader used in the paper. For new data, you can update the code. We used csv files that contained information of train, val, and test sets.