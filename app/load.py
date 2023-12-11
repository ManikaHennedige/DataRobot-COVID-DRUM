import base64
import os
import pandas as pd
import hashlib
import argparse

"""
    This is a sample application script that highlights the preprocessing steps needed for image processing using a DataRobot hosted model.
    This file outlines the necessary functions and processes for single predictions.
    Scripts here are adapted from ipynb notebooks helpfully provided by Tommy
"""

def generate_data(file_dir='train', img_dir='data'):
    """Loads data from local storage and prepares it for transmission over DataRobot. 
    CT-Scan images to be predicted on should be stored in the data/ folder.
    Single .txt file containing the rows of prediction data should also be stored in the data/ folder. Each row in the file should contain the following information:
    {image filename} {actual} {x1} {y1} {x2} {y2}

    Args:
        txt_filename (string): Name of the file containing rows of prediction data and their respective details. This file should be stored in the data/ folder. Defaults to 'data.txt'
    """
    input_data = []
    actual_data = []

    with open('data/' + file_dir + '/data.txt') as f:
        for line in f.readlines():
            fname, cls, x1, y1, x2, y2 = line.strip('\n').split()
            if cls == '1' or cls == '0':
                cls = '0'
            else:
                cls = '1'

            print(fname)
            with open(os.path.join(img_dir, fname), 'rb') as imf:
                imencoded = base64.b64encode(imf.read())
                hash_obj = hashlib.sha256()
                hash_obj.update(str(imencoded).encode('utf-8'))
                id = hash_obj.hexdigest()
                input_data.append([id, imencoded, int(x1), int(y1), int(x2), int(y2)])
                actual_data.append([id, cls])
    
    input_df = pd.DataFrame(input_data, columns=['id', 'img', 'x1', 'y1', 'x2', 'y2'])
    actual_df = pd.DataFrame(actual_data, columns=['id', 'result'])

    input_df.to_csv('data/' + file_dir + '/prepared_data.csv', index=False)
    actual_df.to_csv('data/' + file_dir + '/actual_data.csv', index=False)

def join_files(file_dir):
    """Function to join the created 'prepared_data' and 'actual_data' based on the 'id' column

    Args:
        file_dir (string): folder located inside the 'data' directory containing the files to be joined
    """
    # open the csv files based on the provided file_dir
    file1 = pd.read_csv('data/' + file_dir + '/prepared_data.csv')
    file2 = pd.read_csv('data/' + file_dir + '/actual_data.csv')

    # inner join based on 'id' column
    merged_data = pd.merge(file1, file2, on='id', how='inner')

    # save the tables to a new 'merged_data.csv'
    merged_data.to_csv('data/' + file_dir + '/merged_file.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Generate prediction, actual and merged data in csv format')
    parser.add_argument('-p', '--path', help='Specify the subfolder inside the "app/data" directory containing the relevant "data.txt" file. For example, specifying "test" indicates that the "data.txt" file is located inside the "data/test" folder')
    parser.add_argument('-i', '--imgpath', help='Specify the directory where all the images are stored')
    
    args = parser.parse_args()
    file_dir = args.path
    img_dir = args.imgpath

    print(f"Chosen data.txt directory: data/{file_dir}")
    print(f"Chosen image directory: {img_dir}")


    generate_data(file_dir=file_dir, img_dir=img_dir)
    join_files(file_dir=file_dir)


if __name__ == '__main__':
    main()