import requests
from datetime import datetime
from tqdm import tqdm
import json
import sys
sys.path.append('./LocalFolder/')
import my_base_functions #type: ignore

today_date = datetime.today().strftime('%Y-%m-%d')

def get_tableinfo_from_dst(table_id, endpoint = '/tableinfo'):
    base_url = 'https://api.statbank.dk/v1'
    params = {
        'id': table_id,
        'format': 'JSON'
    }
    print("Fetching table information...")
    response = requests.get(base_url + endpoint, params=params)
    if response.status_code == 200:
        print("Done fetching table information.")
        return response.json()
    else:
        print("Failed to fetch table information.")
        return None
    
def get_bulk_data_from_dst(table_id, request_format = 'BULK'):
    base_url = 'https://api.statbank.dk/v1'
    endpoint = '/data'
    
    # Retrieve variable information
    variables_info = get_tableinfo_from_dst(table_id)['variables']
    variable_ids = [var['id'] for var in variables_info]
    print(variable_ids)
    assert 'OMRÅDE' in variable_ids
    
    # Extract 'OMRÅDE' values directly from the variable information
    omrade_info = next((var for var in variables_info if var['id'] == 'OMRÅDE'), None)
    omrade_values = [item['id'] for item in omrade_info['values']] if omrade_info else []
    
    all_data = []
    # Chunk the OMRÅDE values for batched requests
    chunk_size = 2
    omrade_chunks = [omrade_values[i:i+chunk_size] for i in range(0, len(omrade_values), chunk_size)]
    
    total_chunks = len(omrade_chunks)
    print(f"Fetching bulk data in {total_chunks} chunks...")
    for idx, omrade_chunk in tqdm(enumerate(omrade_chunks), desc="Fetching Data Chunks", total=total_chunks):
    
        variables_list = []
        for variable_id in variable_ids:
            values = omrade_chunk if variable_id == 'OMRÅDE' else ['*']
            variable_entry = {
                "code": variable_id,
                "values": values
            }
            variables_list.append(variable_entry)
                
        params = {
            'table': table_id,
            'format': request_format,
            'variables': variables_list
        }
        
        response = requests.post(base_url + endpoint, json=params)
    
        if response.status_code == 200:
            all_data.append(response.text)
        else:
            print(f"Failed request for OMRÅDE chunk: {omrade_chunk}")
            print(response)
    
    combined_data = "\\n".join(all_data)
    return combined_data

if __name__ in '__main__':
    for table_name in ["AUS08"]:
        try:
            data = get_bulk_data_from_dst(table_name)
            my_base_functions.save_to_csv(data, f"{table_name}_export_{today_date}")
        except:
            pass


"""
Brugte tabeller:
Gennemsnitsalder: GALDER
Anmeldte forbrydelse: STRAF11 (22 er sejere med rummer mindre data) BENYT MED 'BULK'!!!
Middellevealder: HISBK
Disponibel indkomst: INDKP106
Folketal: FOLK1C
Formue i fast ejendom: EJERFOF1
Arbejdssteder: ERHV5
Employment: AUS08
Samlet byggeaktivitet: BYGV11
Arealer for kommuner: ARE207
Personbeskatning: PSKAT
Ejendomsbeskatning: EJDSK1      eller EJDSK2
Uddannelsesaktivitet: UDDAKT10
Folketal: FOLK1A

"EJERFOF1", "ERHV5", "BYGV11", "ARE207", "PSKAT", "EJDSK1", "EJDSK2", "UDDAKT10", "FOLK1A", "FOLK1C"

Find tabel: https://www.statistikbanken.dk/statbank5a/default.asp?w=1440
Find tabel efter kommune: https://www.dst.dk/da/Statistik/kommunekort/statistikbanktabeller-paa-kommuneniveau

"""




