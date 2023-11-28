## DRUM-Compatible Model and Environment

## Introduction

DRUM, or DataRobot User Models, provide a way for users to test custom models and environments before they are uploaded and deployed to the DataRobot platform. This repository comprises 3 main folders, with each folder performing its' own distinct function.

1. 'environment' folder

    This folder contains the Docker context files that are required to create the Docker environment used to run the model in.

1. 'model' folder

    This folder contains files required to create a DR-compatible model.

1. 'app' folder

    This folder contains files for the running of a couple of CLI interface application to generate prediction files and perform predictions using the created environment and model.


## Folder Details

### 'environment' folder

This folder was obtained through copying a folder from [DataRobot's DRUM Repository](https://github.com/datarobot/datarobot-user-models/tree/master/public_dropin_environments/python3_keras)

The function and purpose of the files are outlined below:

1. Dockerfile

    Contains the necessary prerequisites (such as Python version) that are needed to create the environment. Make amendments as necessary.

1. dr_requirements.txt

    Contains the requirements for the base DataRobot environment. If the Dockerfile's initial import in the Dockerfile does not need to be changed, chances are the dependencies in this file won't need to be changed either.

1. requirements.txt

    Contains the requirements for running the custom user-defined environment. Any packages/dependencies that are used by the 'model' folder should be specified here


### 'model' folder

This folder was generated by running:

```drum new model --language python --code-dir model```

1. utils

    Folder containing optional utility files and scripts. Code scripts and other files can be placed here. This folder currently only contains a single 'functions.py' script, which contain functions that are called in 'custom.py'

1. custom.py

    Script that executes when a custom model is run via DRUM/DataRobot. The functions execute in sequence if they are not commented out. The template for this file is autogenerated using the above command.

1. phase-1.h5

    Model weights that are loaded into DRUM/DataRobot using the custom.py file.

### 'app' folder

This folder contains two CLI applications that can be run to generate data, then perform predictions with the DataRobot/DRUM API.

1. 'data' folder

    This directory contains several folders, with each folder containing an example instance of data that can be predicted by DataRobot/DRUM. Only the 'data.txt' file is required as a prerequisite. The other csv files are generated by the script 'load.py'.

1. load.py

    This script is customized for use with the COVID CT-Scan project. It allows users to generate 3 csv files based on a 'data.txt' file, which can then be uploaded to DataRobot for testing and data quality metric tracking.

1. predict.py

    This script is adapted from the official DR script, and is more generally usable. It allows real-time predictions to be made on the DataRobot or DRUM platform.

## Prerequisites

Note that DRUM requires a Linux environment to run. If on Windows, use Windows Subsystem for Linux (WSL)

### Installing DRUM
DRUM should be installed in the global Python environment (within Linux/WSL on Windows):

```pip install datarobot-drum```

### Creating a virtual environment (optional)
A virtual environment may be created after installing DRUM:

```virtualenv wsl_env```

```source wsl_env/bin/activate```

Scripts run in this virtual environment will usually only come from the 'app' folder - dependencies in the 'environment' folder do not necessarily need to be installed here. Install requirements located in the 'app' folder

```pip install app/requirements.txt```

## Use

### Loading data into compatible csv files

The 'load.py' file contains scripts and a corresponding CLI application for users to generate a csv file for prediction. Before running the script, ensure that the following conditions are met:

1. data.txt file

    A data.txt file containing prediction details should be created and placed into a named folder inside the 'app/data' directory. The name of this folder will be required to run the CLI applications.
    
    This file should contain at least one line of data, with each line containing the following information in the format specified below:

    ```{image_file_name} {result class} {x1} {y1} {x2} {y2}```

    An example line in the file could be:

    ```pred_img.png 0 15 100 200 400```

    - The result class (indicating whether the image contains a COVID positive patient or not) should be either 0, 1 or 2, similar to the original dataset

1. image folder

    All the images referenced in the above created data.txt file should be available in a certain folder. There is no specific guideline for where this folder should be located. The absolute path to the folder will be required to run the 'load.py' CLI application.

Run the load.py CLI application to generate the necessary files for prediction:

```python load.py -p [name of folder containing 'data.txt'] -i [name of folder containing images]```

For example, if 'data.txt' is stored inside 'data/test1' folder, and images are stored inside 'C:\Users\SGH\COVID_Data\2A_images', the command would be:

```python load.py -p 'test1' -i 'C:\Users\SGH\COVID_Data\2A_images'```

Three files will be generated inside the 'test1' folder:

1. prepared_data.csv

    This contains the data for prediction on the DataRobot/DRUM server

1. actuals.csv

    This contains actuals for upload to DataRobot - this file is required for accuracy metrics to be generated.

1. merged_file.csv

    This contains the merged data from both 'prepared_data.csv' and 'actuals.csv'.
    
### Running real time predictions on the DRUM server

Now that the required data has been generated, we may perform predictions with this data using the custom model in the custom environment.

1. Run the model using DRUM, using the model specified in the 'model/' directory and the environment specified in the 'environment/' directory. Run the command below from the root project directory:

    ```drum server --code-dir model/ --target-type binary --positive-class-label '1' --negative-class-label '0' --address localhost:6789 --docker environment```

1. Make a prediction on the DRUM server by running the following command from inside the 'app' folder:

    ```python predict.py -m [multiprocessing mode of prediction (either 'single' or 'multi')] -p [name of folder containing the generated 'prepared_data.csv file] -dev```

    For example, if the 'prepared_data.csv' file is stored inside 'data/test1', and we choose to run this model in a singlethreaded manner, the command would be:

    ```python predict.py -m single -p data/test1/prepared_data.csv -dev```

### Making real time predictions on the model deployed on the DataRobot server

1. The environment and model must be uploaded and deployed on the DataRobot server before we may predict on the server. For instructions on uploading to DataRobot, refer to the [next section](#uploading-to-datarobot)

1. Make a copy of the .env-sample file, renaming it to '.env'. Make sure that the endpoint, deployment ID and API token is provided in this file.

1. Run the following command from inside the 'app' folder:

    ```python predict.py -m [multiprocessing mode of prediction (either 'single' or 'multi')] -p [name of folder containing the generated 'prepared_data.csv file]```

    For example, if the 'prepared_data.csv' file is stored inside 'data/test1', and we choose to run this model in a singlethreaded manner, the command would be:

    ```python predict.py -m single -p data/test1/prepared_data.csv```


## Uploading to DataRobot

Once the model and environment have been tested using the local DRUM prediction server, the environment and model may then be uploaded to DataRobot. The uploading process is quite simple, and detailed instructions are available on the DataRobot server website.

### Creating a new custom environment in DataRobot

Create a new custom environment from the DataRobot UI's 'Custom environments' area, then drop the 'environment.zip' file into the 'Docker context' field. The environment will be created using the context provided. It is also possible to upload a tarball containing the docker environment, but the context would need to be specified too

### Creating a new custom model in DataRobot

Create a new model from the DataRobot UI's 'custom model workshop', making sure to choose the target type as unstructured. Use the 'file upload' feature instead of the folder upload feature to ensure that the custom.py and model.h5 files are accessible from the root directory. The 'folder upload' feature can be used for dependencies (e.g. in this case the 'utils' folder can be uploaded via this method).

Test the model, then deploy.