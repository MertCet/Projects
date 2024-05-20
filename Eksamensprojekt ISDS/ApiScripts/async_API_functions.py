import asyncio
import httpx
import sys
sys.path.append('./LocalFiles/')
import my_secrets #type: ignore


# Authentication details
username = my_secrets.tjenestebruger1_usr
password = my_secrets.tjenestebruger1_pass

async def get_BBR_grund_data_from_BFE(BFE_number, session):
    """ #? Variables returned:
    datafordelerOpdateringstid, adresseIdentificerer, enh020EnhedensAnvendelse, 
    enh023Boligtype, enh026EnhedensSamledeAreal, enh027ArealTilBeboelse, 
    enh031AntalVærelser, enh065AntalVandskylledeToiletter, enh066AntalBadeværelser, 
    etage, id_lokalId, id_namespace, kommunekode, opgang, registreringFra, registreringTil, bygning
    """
    url = "https://services.datafordeler.dk/BBR/BBRPublic/1/REST/grund"
    
    # Parameters for the request
    params = {
        "BFENummer": BFE_number,
        "username": username,
        "password": password,
        "format": "JSON"
    }

    # Make the request
    response = await session.get(url, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Request failed with status code: {response.status_code}")
        return {"error": f"Request failed with status code: {response.status_code}"}
    
async def get_BBR_bygning_data_from_grund_id(grund_id, session):
    url = "https://services.datafordeler.dk/BBR/BBRPublic/1/rest/bygning"
    params = {
        "grund": grund_id,
        "username": username,
        "password": password
    }
    response = await session.get(url, params=params)
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Request failed with status code: {response.status_code}")
        return {"error": f"Request failed with status code: {response.status_code}"}

async def get_BBR_enhed_data_from_bygning_id(bygningid, session):
    url = "https://services.datafordeler.dk/BBR/BBRPublic/1/rest/enhed"
    params = {
        "bygning": bygningid,
        "username": username,
        "password": password
    }
    response = await session.get(url, params=params)
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Request failed with status code: {response.status_code}")
        return {"error": f"Request failed with status code: {response.status_code}"}

async def get_EBR_data_from_BFE(BFEnr, session):
    url = "https://services.datafordeler.dk/EBR/Ejendomsbeliggenhed/1/rest/Ejendomsbeliggenhed?"
    params = {
        "BFEnr": BFEnr,
        "username": username,
        "password": password
    }
    response = await session.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Request failed with status code: {response.status_code}")
        return {"error": f"Request failed with status code: {response.status_code}"}


BygAnvendelse_kode_list = [120, 121, 122, 131, 132, 140, 150, 540]
BygAnvendelse_kode_text = ["Fritliggende enfamiliehus", "Sammenbygget enfamiliehus", "Fritliggende enfamiliehus i tæt-lav bebyggelse", 
                      "Række-, kæde- og klyngehus", "Dobbelthus", "Etagebolig-bygning, flerfamiliehus eller to-familiehus", "Kollegium", "Kolonihavehus"]

async def get_vur_by_BFE(BFEnummer, session):
    url = "https://services.datafordeler.dk/Ejendomsvurdering/Ejendomsvurdering/1/rest/HentEjendomsvurderingerForBFE"
    username = my_secrets.tjenestebruger1_usr
    password = my_secrets.tjenestebruger1_pass
    params = {
        "BFEnummer": BFEnummer,
        "username": username,
        "password": password
    }
    headers = {
        "Accept": "application/json"
    }
    response = await session.get(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Request failed with status code: {response.status_code}")
        return {"error": f"Request failed with status code: {response.status_code}"}

async def get_MAT_from_BFE(BFE_nummer, session):
    url = "https://services.datafordeler.dk/Matriklen2/Matrikel/1.0.0/rest/BestemtFastEjendom"
    params = {
        "BFEnr": BFE_nummer,
        "username": username,
        "password": password
    }
    response = await session.get(url, params=params)
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Request failed with status code: {response.status_code}")
        return {"error": f"Request failed with status code: {response.status_code}"}


async def example_test(data):
        from datetime import datetime
        BygAnvendelse_kode_list = ['120', '121', '122', '140', '540']
        BygAnvendelse_kode_text = ["Fritliggende enfamiliehus", "Sammenbygget enfamiliehus", "Fritliggende enfamiliehus i tæt-lav bebyggelse", 
                            "Række-, kæde- og klyngehus", "Kolonihavehus"]
        relevant_items = [item for item in data if item.get('byg021BygningensAnvendelse') in BygAnvendelse_kode_list]
        # if there are no relevant items, return
        if len(relevant_items) == 0:
            return
        # if there are relevant items, group by Bygningsnummer
        grouped = {}
        for item in relevant_items:
            bygningsnummer = item.get('byg007Bygningsnummer')
            if bygningsnummer is None:
                continue
            if bygningsnummer in grouped:
                grouped[bygningsnummer].append(item)
            else:
                grouped[bygningsnummer] = [item]

        list_to_return = []
        for bygningsnummer, items in grouped.items():
            sorted_items = sorted(items, key=lambda x: datetime.fromisoformat(x['virkningFra']), reverse=True)
            
            newest_item = sorted_items[0]
            anvendelse_kode_text = BygAnvendelse_kode_text[BygAnvendelse_kode_list.index(newest_item.get("byg021BygningensAnvendelse"))]
            
            # Add attributes of the newest item as tuple elements
            newest_item_tuple = (
                newest_item.get("id_lokalId"),
                newest_item.get("byg039BygningensSamledeBoligAreal"),
                newest_item.get("byg041BebyggetAreal"),
                newest_item.get("byg042ArealIndbyggetGarage"),
                newest_item.get("byg026Opførelsesår"),
                newest_item.get("byg027OmTilbygningsår"),
                anvendelse_kode_text,
                newest_item.get("byg404Koordinat"),
                newest_item.get("byg406Koordinatsystem"),
            )
            
            older_items_list = []
            for older_item in sorted_items[1:]:  # Skip the newest item
                older_item_attributes = {
                    'id_lokalId': older_item.get("id_lokalId"),
                    'byg039BygningensSamledeBoligAreal': older_item.get("byg039BygningensSamledeBoligAreal"),
                    'byg041BebyggetAreal': older_item.get("byg041BebyggetAreal"),
                    'byg042ArealIndbyggetGarage': older_item.get("byg042ArealIndbyggetGarage"),
                    'byg026Opførelsesår': older_item.get("byg026Opførelsesår"),
                    'byg027OmTilbygningsår': older_item.get("byg027OmTilbygningsår"),
                    'anvendelse_kode_text': anvendelse_kode_text
                }
                older_items_list.append(older_item_attributes)
            
            # Append older items as the last element in the tuple
            newest_item_tuple_with_older_items = newest_item_tuple + (older_items_list,)
            
            list_to_return.append(newest_item_tuple_with_older_items)
        
        return list_to_return


async def example_func():
    async with httpx.AsyncClient(timeout=None) as session:
        bfe = "8388897"
        data = await get_BBR_grund_data_from_BFE(bfe, session) 
        print(data[0].get("id_lokalId"))
        data = await get_BBR_bygning_data_from_grund_id("11c163f0-a63e-4a85-be8f-b626e628dca9", session)
        my_base_functions.save_to_json(data, "temp_data")

if __name__ in '__main__':
    import my_base_functions  #type: ignore
    from my_base_functions import extract_attribute #type: ignore
    asyncio.run(example_func())