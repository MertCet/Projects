import requests
import sys
sys.path.append('./LocalFolder/')
import my_base_functions #type: ignore
from datetime import datetime

today_date = datetime.today().strftime('%Y-%m-%d')

def get_table_metadata(tableID, lang="en"):
    base_url = "https://api.statbank.dk/v1"
    endpoint = f"/tableinfo/{tableID}?lang={lang}"
    full_url = base_url + endpoint

    response = requests.get(full_url)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

if __name__ in '__main__':
    # Example usage:
    table_id = "FOLK1C"  # Replace with the desired table ID
    metadata = get_table_metadata(table_id)

    # Save to outputs folder
    my_base_functions.save_to_json(metadata, f"{table_id}_metadata_export_{today_date}")
