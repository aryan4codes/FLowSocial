

## Backend Setup : 

This project is managed using [Poetry](https://python-poetry.org/), a tool for dependency management and packaging in Python. This README provides a guide on how to set up, install dependencies, and use Poetry in this project.

### Data : Open Images Dataset - Google

## Prerequisites
- Python 3.11 (or the required version specified in `pyproject.toml`)
- Poetry (install via `pip install poetry` or follow [Poetry's installation guide](https://python-poetry.org/docs/#installation))

## Setting Up the Project

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
cd project
```

### 2. Set Up Poetry Environment
Ensure you are using the correct Python version:
```bash
poetry env use python3.11 
```

or 
```
poetry env use {your python bin path} 
```

Find that using "where python" in Git bash 


### 3. Install Project Dependencies
Run the following command to install all dependencies as specified in `pyproject.toml`:
```bash
poetry install
```

This will create a virtual environment and install all required packages.

### 4. Activate the Poetry Shell (Optional)
To activate the virtual environment created by Poetry, run:
```bash
poetry shell
```

This will allow you to run Python commands and scripts within the context of the project's virtual environment.
