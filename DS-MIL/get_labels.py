import requests
import json
import re
import pandas as pd
import os 
from typing import List


def get_file_id(file_name):
    fields = [
    "file_id",
    ]

    fields = ",".join(fields)

    cases_endpt = "https://api.gdc.cancer.gov/files"

    filters = {
    "op": "in",
    "content":{
        "field": "file_name",
        "value": file_name
        }
    }

    # With a GET request, the filters parameter needs to be converted
    # from a dictionary to JSON-formatted string

    params = {
    "filters": json.dumps(filters),
    "fields": fields,
    "format": "json",
    #"size": "100"
    }

    response = requests.get(cases_endpt, params = params)

    d = response.json() 
    return d['data']["hits"][0]['file_id']



def get_csv_file(file_id : List['str']) -> None:
    fields = [
    "file_name",
    "cases.diagnoses.primary_diagnosis",
    "cases.case_id",
    "file_id"
    ]

    fields = ",".join(fields)

    List = []
    for file in file_id:
        files_endpt = "https://api.gdc.cancer.gov/files/" + file

    # A POST is used, so the filter parameters can be passed directly as a Dict object.
        params = {
        #"filters": filters,
            "fields": fields,
            "format": "json",
            }

        # The parameters are passed to 'json' rather than 'params' in this case
        response = requests.post(files_endpt, headers = {"Content-Type": "application/json"}, json = params)

        #print(json.dumps(response.json(), indent=2))
        d = response.json()

    #diagnosis :-
        diagnosis = d['data']['cases'][0]['diagnoses'][0]['primary_diagnosis']
        caseID = d['data']['cases'][0]['case_id']
        fileID = d['data']['file_id']
        #print(diagnosis)
    
        List.append([caseID, fileID, diagnosis])
        #print(List)
    
    column_names = ['caseID', 'fileID', 'diagnosis']
    df = pd.DataFrame(List, columns=column_names)
    
    csv_file_path = '/scratch/shubham.ojha/LUNG.csv'
    df.to_csv(csv_file_path, index=False)




if __name__ == '__main__':
    WSI_list = os.listdir('/scratch/shubham.ojha/LUNG_DATA_DIRECTORY')

    file_id_list = []
    for wsi in WSI_list:
        file_id = get_file_id(wsi)
        file_id_list.append(file_id)

    get_csv_file(file_id_list)

    