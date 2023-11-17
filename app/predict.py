import pandas as pd
from multiprocessing.pool import ThreadPool
import requests
import argparse
from load import load_data

# below should be uncommented for production

def rtpred(pred_url, data, token):
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
    # single threaded for loop scoring
    for df_chunk in df:
        pred_resp = rtpred(pred_url, df_chunk.to_csv(), token)
        print(pred_resp.text)

def multi_threaded_batch_processing(df, pred_url, token):
    # multi-threaded for loop scoring
    threads = []
    pool = ThreadPool()

    for i, df_chunk in enumerate(df):
        async_result = pool.apply_async(rtpred, (pred_url, df_chunk.to_csv(), token))
        threads.append(async_result)
        print(async_result)

    for thread in threads:
        print(thread.get())

def main():
    parser = argparse.ArgumentParser(description='Example program with flag arguments')
    parser.add_argument('-m', '--mode', default='single', help='Specify the real-time prediction mode. Specify "single" for single-threaded processing, or "multi" for multithreaded processing')
    parser.add_argument('-p', '--path', default='data/prepared_data.txt', help='Specify the path to the .txt file containing the data')
    parser.add_argument('-d', '--development', action='store_true', help='Specify whether development mode should be enabled')
    parser.add_argument('-c', '--chunksize', default='1', help='Specify the size of the chunk to be used such that each chunk would be less than 50MB')

    args = parser.parse_args()

    # Access the parsed arguments
    mode = args.mode
    path = args.path
    development = args.development
    chunksize = args.chunksize

    # Print the arguments
    print(f"Mode: {mode}")
    print(f"Path: {path}")
    print(f"Development: {development}")
    print(f"Chunk size: {chunksize}")

    token = 'NjUzZjYyNjRhZTIzYTc4NWU0MTg0YmQ4OkdJaThFU25NbWg1bmtFSENORVB3TkpVUlFvZG4vT01QQmh4NmNhdy9Nam89'
    endpoint = 'app.imda-tal-ent.sg.datarobot.com'
    deployment_id = "653f753308ddc534ae184bf9"

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