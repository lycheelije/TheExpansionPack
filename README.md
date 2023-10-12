SETUP
-----
Consider using a Virtual Environment (via Integrated Terminal on this Project)

## Path EG: PS C:\Users\pc-user\OneDrive - University of Technology Sydney\Github\TheExpansionPack>

##### Windows:
## Install venv package if missing (included in Py3.3>)
.\TheExpandsionPack> py -m pip install --user virtualenv

## Generate venv ()
.\TheExpandsionPack> python -m venv venv
.\TheExpandsionPack>.\venv\Scripts\activate

## Install Project Requirements
When activated, you will notice the path cahnge to include (venv). This will show that you are in the venv for managing our project-specific dependencies.

Example:(venv) PS C:\Users\pc-user\OneDrive - University of Technology Sydney\Github\TheExpansionPack>

## Ensure the Python Interpreter is set to the venv
Open Visual Code.
Go to 'View' on the toolbar.
Click on 'Command Palette...' (or press CTRL + SHIFT + P on Windows).
Search for: 'Python Select Interpreter'.
Click on: 'Enter interpreter path...' followed by 'Find...'.
Go to your (venv) scripts folder: project_folder/venv/Scripts.
Select "pythonw.exe" inside of your project_folder/venv/Scripts folder.

## Install 'Requests' - Allows for installing specific versions of a package (may be included already)
.\TheExpandsionPack> py -m pip install requests

## Installs all dependencies in the requirements.txt file 
.\TheExpansionPack> py -m pip install -r requirements.txt

### Note issue with CUDNN using this flow - version not found - cutting out ML dependencies for now

## Usage
.\TheExpandsionPack> python app.py
