import base64
import os
import pandas as pd

"""
    This is a sample application script that highlights the preprocessing steps needed for image processing using a DataRobot hosted model.
    This file outlines the necessary functions and processes for single predictions.
    Scripts here are adapted from ipynb notebooks helpfully provided by Tommy
"""

def load_data(txt_filename='data.txt'):
    """Loads data from local storage and prepares it for transmission over DataRobot. 
    CT-Scan images to be predicted on should be stored in the data/ folder.
    Single .txt file containing the rows of prediction data should also be stored in the data/ folder. Each row in the file should contain the following information:
    {image filename} {actual} {x1} {y1} {x2} {y2}

    Args:
        txt_filename (string): Name of the file containing rows of prediction data and their respective details. This file should be stored in the data/ folder. Defaults to 'data.txt'
    """
    input_data = []

    with open(os.path.join('data', txt_filename)) as f:
        for line in f.readlines():
            fname, cls, x1, y1, x2, y2 = line.strip('\n').split()
            print(fname)
            with open(f'data/{fname}', 'rb') as imf:
                imencoded = base64.b64encode(imf.read())
                input_data.append([imencoded, int(x1), int(y1), int(x2), int(y2)])
    
    df = pd.DataFrame(input_data, columns=['img', 'x1', 'y1', 'x2', 'y2'])

    df.to_csv('data/prepared_data.txt', index=False)

load_data()