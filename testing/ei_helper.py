"""
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

import requests
import pandas as pd

# --------------------------------------------------------------------------------------------------------------

url = "https://studio.edgeimpulse.com/v1"
url_classify_job_result             = url + "/api/{projectId}/classify/all/result"
url_raw_data_list_samples           = url + "/api/{projectId}/raw-data"

# --------------------------------------------------------------------------------------------------------------

def classify_job_request(x_api_key, project_id):
    """Get classify job result, containing the result for the complete testing dataset.
    
    ref.: https://docs.edgeimpulse.com/reference/edge-impulse-api/classify/classify-job-result
    
    Parameters
    ----------
    x_api_key: str
        edge impulse API key
    project_id: str
        project ID

    """
    
    url = url_classify_job_result.replace("{projectId}", project_id)

    headers = {
        "Accept": "application/json",
        "x-api-key": x_api_key
    }

    return requests.request("GET", url, headers=headers)


def raw_data_list_samples(x_api_key, project_id, category):
    """Retrieve all raw data by category.
        
    Parameters
    ----------
    x_api_key: str
        edge impulse API key
    project_id: str
        project ID
    category: str
        The categories to retrieve data from: "testing", "training", "anomaly"
    
    """
  
    url = url_raw_data_list_samples.replace("{projectId}", project_id)

    querystring = {
        "category": category
        }

    headers = {
        "Accept": "application/json",
        "x-api-key": x_api_key
    }
    
    return requests.request("GET", url, headers=headers, params=querystring)


def model_testing_result_tbl(x_api_key, project_id):
    """ Create test results table
    """

    category = "testing"

    # Get classify job result, containing the result for the complete testing dataset.
    response_classify_job_request = classify_job_request(x_api_key, project_id)
    # Retrieve all raw data by category = testing.
    response_raw_data_list_samples = raw_data_list_samples(x_api_key, project_id, category)


    total_nmb_samples = len(response_classify_job_request.json()["result"])

    # table: ['sampleId', 'result']
    data_response_classify_job_request = []
    for sample in range(total_nmb_samples):
        sampleId = response_classify_job_request.json()["result"][sample]["sampleId"]
        value    = response_classify_job_request.json()["result"][sample]["classifications"][0]["result"][0]["value"]
        
        data_response_classify_job_request.append([sampleId, float(value)])

    df_response_classify_job_request = pd.DataFrame(data_response_classify_job_request, columns = ['sampleId', 'result'])

    # table: ['sampleId', 'filename', 'expected result']
    data_response_raw_data_list_samples = []
    for sample in range(total_nmb_samples):
        sampleId = response_raw_data_list_samples.json()["samples"][sample]["id"]
        filename = response_raw_data_list_samples.json()["samples"][sample]["filename"]
        label    = response_raw_data_list_samples.json()["samples"][sample]["label"]
        
        data_response_raw_data_list_samples.append([sampleId, filename, float(label)] )

    df_response_raw_data_list_samples = pd.DataFrame(data_response_raw_data_list_samples, columns = ['sampleId', 'filename', 'expected result'])

    # table: ['sampleId', 'filename', 'expected result', 'result']
    columns = ['sampleId', 'filename', 'expected result', 'result']
    table = df_response_classify_job_request.set_index('sampleId').combine_first(df_response_raw_data_list_samples.set_index('sampleId')).reset_index()[columns]

    return table