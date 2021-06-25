# ResearchProject
codebase for research project

To use the virtual environment execute:

```
python3 -m venv . 
source bin/activate
python3 -m pip install -r requirements.txt
```

If you get the an error: 
```
module 'keras.utils.generic_utils' has no attribute 'populate_dict_with_module_objects'
```
Try uninstalling keras-nightly and reinstalling tensorflow, for some reason this fixes it. 
