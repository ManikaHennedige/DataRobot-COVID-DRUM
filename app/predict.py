import pandas as pd
from multiprocessing.pool import ThreadPool
import requests
import argparse
from dotenv import dotenv_values



def rtpred(pred_url, data, token):
    """Function calling the endpoint and sending the request to perform a real time prediction on provided data

    Args:
        pred_url (string): URL endpoint for the DataRobot deployment
        data (pd.DataFrame): DataFrame containing test data to be predicted on.
        token (string): API token required for calling the DataRobot API

    Returns:
        Request: Request object containing prediction results
    """
    response = requests.post(
        url=pred_url, 
        headers={
            'Authorization': f'bearer {token}',
            'Content-Type': 'text/csv; charset=UTF-8'
        },
        data=data
    )
    return response

def single_threaded_batch_processing(df, pred_url, token):
    """Run the real-time prediction on a DataRobot model deployment on a single threaded instance

    Args:
        df (pd.DataFrame): DataFrame containing the test data to be predicted on
        pred_url (string): URL endpoint for the DataRobot deployment
        token (string): API token required for calling the DataRobot API
    """
    for df_chunk in df:
        pred_resp = rtpred(pred_url, df_chunk.to_csv(), token)
        print(pred_resp.text)

def multi_threaded_batch_processing(df, pred_url, token):
    """Run the real-time prediction on a DataRobot model deployment using multiple threads for improved efficiency

    Args:
        df (pd.DataFrame): DataFrame containing the test data to be predicted on
        pred_url (string): URL endpoint for the DataRobot deployment
        token (string): API token required for calling the DataRobot API
    """
    threads = []
    pool = ThreadPool()

    for _, df_chunk in enumerate(df):
        async_result = pool.apply_async(rtpred, (pred_url, df_chunk.to_csv(), token))
        threads.append(async_result)

    for thread in threads:
        # responses from each thread
        print(thread.get().text)

def main():
    parser = argparse.ArgumentParser(description='Predict on models deployed on DataRobot or DRUM')
    parser.add_argument('-m', '--mode', default='single', help='Specify the real-time prediction mode. Specify "single" for single-threaded processing, or "multi" for multithreaded processing')
    parser.add_argument('-p', '--path', default='data/test/prepared_data.csv', help='Specify the path to the csv file containing the data')
    parser.add_argument('-dev', '--development', action='store_true', help='Flag indicating that development mode should be enabled. Calling this flag would set the endpoint URL to a local instance, while excluding this flag would set the endpoint to a live DataRobot deployment')
    parser.add_argument('-c', '--chunksize', default='1', help='Specify the size of the chunk to be used such that each chunk would be less than 50MB')

    args = parser.parse_args()

    mode = args.mode
    path = args.path
    development = args.development
    chunksize = args.chunksize

    config = dotenv_values(".env")

    token = config.get('TOKEN')
    endpoint = config.get('ENDPOINT')
    deployment_id = config.get('DEPLOYMENT_ID')

    # print the input & loaded arguments for verification
    print(f"Mode: {mode}")
    print(f"Path: {path}")
    print(f"Development Mode: {development}")
    print(f"Chunk Size: {chunksize}")
    print(f"Endpoint: {endpoint}")
    print(f"Deployment ID: {deployment_id}")
    print(f"API Token Provided: {token != '' and token != None}")



    # choose a prediction url based on whether we are in development mode
    if development:
        pred_url = 'http://localhost:6789/predict'
    else:
        pred_url = f'https://{endpoint}/predApi/v1.0/deployments/{deployment_id}/predictions'

    
    df = pd.read_csv(path, chunksize=int(chunksize))

    if mode == 'single':
        single_threaded_batch_processing(df, pred_url, token)
    elif mode == 'multi':
        multi_threaded_batch_processing(df, pred_url, token)

if __name__ == "__main__":
    main()