""" JSON file upload to Edge Impulse Studio

MIT License

Copyright (c) 2023 - J.R.Verbiest

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import os
import argparse
import json
import hashlib
import hmac
import requests

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def upload_data(API_KEY, HMAC_KEY, x_file_name, x_data, x_label, x_category,):
    """
    Parameters
    ----------
    HMAC_KEY: str
        Edge Impulse HMAC key
        
    API_KEY: str
        Edge Impulse API key
        
    x_file_name: str
        filename

    x_data: json
        data

    x_label: str
        label

    x_category: str
        category: training or testing
    """
    
    # encode in JSON
    encoded = json.dumps(x_data)

    # sign message
    signature = hmac.new(bytes(HMAC_KEY, 'utf-8'), msg = encoded.encode('utf-8'), digestmod = hashlib.sha256).hexdigest()

    # set the signature again in the message, and encode again
    x_data['signature'] = signature
    encoded = json.dumps(x_data)

    # and upload the file
    res = requests.post(url='https://ingestion.edgeimpulse.com/api/'+x_category+'/data',
                        data=encoded,
                        headers={
                            'Content-Type': 'application/json',
                            'x-file-name': x_file_name,
                            'x-label': str(x_label), 
                            'x-api-key': API_KEY
                        })

    if (res.status_code == 200):
        print('Uploaded file to Edge Impulse', x_category, res.status_code, res.content)
    else:
        print('Failed to upload file to Edge Impulse', x_category, res.status_code, res.content)


def run(args):

    EI_API_KEY=args.ei_api_key
    logger.info("EI_API_KEY: %s", EI_API_KEY)
    
    EI_HMAC_KEY=args.ei_hmac_key
    logger.info("EI_HMAC_KEY: %s", EI_HMAC_KEY)

    data_loc=args.data_location
    data_loc=data_loc 
    logger.info("data location: %s", data_loc)

    files = []
    for file in os.listdir(data_loc):
        if file.endswith(".json"):
            files.append(file)


    # read all json from the the data_location into a file list
    # example file: Sub_<..>-strideLengthcm-124.json    
    for file in files:
        json_file = open(os.path.join(data_loc,file),"r")
        data = json.loads(json_file.read())
        json_file.close

        x_label = file.split(".json")[0].split("-")[-1]     
        x_category = data_loc.split("/")[-2].split("-")[-1]                

        logger.info("x_label: %s | x_category: %s | Upload file: %s" %(x_label, x_category, file))
        upload_data(EI_API_KEY, EI_HMAC_KEY, file, data, x_label, x_category)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="upload data to Edge Impulse platform.")

    parser.add_argument("--data_location", type=str, help="location (directory) dataset (*.json files).", required=True)
    parser.add_argument("--ei_api_key", type=str, help="ei api key", required=True)
    parser.add_argument("--ei_hmac_key", type=str, help="ei hmac key", required=True)
    args = parser.parse_args()

    run(args)