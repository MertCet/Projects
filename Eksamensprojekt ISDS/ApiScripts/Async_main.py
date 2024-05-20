
import pandas as pd
import json
import pandas as pd
import os
from datetime import datetime
import re
import asyncio
import httpx
import json
import logging
from aiolimiter import AsyncLimiter
import progressbar
import pandas as pd
from time import perf_counter
import numpy as np
import time
import sys
import async_API_functions as my_functions
sys.path.append('./LocalFolder/')
import my_base_functions #type: ignore
from my_base_functions import extract_attribute #type: ignore


# Constants
MAX_ATTEMPTS = 5
SEMAPHORE_LIMIT = 100 # Max number of simultaneous calls
GLOBAL_QUEUE_SIZE = 100000
GLOBAL_CACHED_COUNTER_SIZE = 2000
rate_limiter = AsyncLimiter(1750) # Max number of calls per minute
cache_count = 0
cache_count_failed = 0

# Global variables
response_queue = asyncio.Queue()
failed_queue = asyncio.Queue()
continue_flag = True
call_count = 0 # call count variable
call_timer = perf_counter() # call timer variable

# Import cache
try:
    with open(f'./LocalFiles/data_cache.json') as f:
        cache = json.load(f)
    with open(f'./LocalFiles/data_cache_failed.json') as f:
        cache_failed = json.load(f)
except:
    cache = {}
    cache_failed ={}

cache_list = [item for item in cache]
cached_ids = [item['BFE_nummer'] for item in cache_list]

cache_list_failed = cache_failed.get('BFE_nummer')
if cache_list_failed:
    cached_ids_failed = [item for item in cache_list_failed]
else:
    cache_list_failed = []
    cached_ids_failed = []
print(f"Cache size: S{len(cached_ids)}:F{len(cached_ids_failed)}")

# Load data function
def load_and_filter_BFE_numbers():
    print("Loading data")
    local_file_path = r".\LocalFiles\BFE_numbers.json"
    with open(local_file_path, 'r', encoding='utf-8') as file:
        BFEnummerjson = json.load(file)
    BFEnummerList = BFEnummerjson['BFE_numbers']
    # remove dublicates
    BFEnummerList = set(BFEnummerList)
    BFEnummerList = list(BFEnummerList) 
    print("done")
    return BFEnummerList

### ! DATA FETCHING HELPER FUNCTIONS
#? Data handling functions

def filter_vur_data(data, target_date_str = "2012-06-30"):
    # Convert target date to datetime object
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    
    # Extract the required attributes for items with ændringDato after target_date
    extracted_data = []
    for item in data:
        if datetime.strptime(item["ændringDato"].split("T")[0], "%Y-%m-%d") > target_date:
            extracted_item = {
                "ejendomværdiBeløb": item["ejendomværdiBeløb"],
                "grundværdiBeløb": item["grundværdiBeløb"],
                "vurderetAreal": item["vurderetAreal"],
                "ændringDato": item["ændringDato"],
                "antalMedvirk": item["antalMedvurderedeLejligheder"]
            }
            
            # Extract the "beløb" value where "tekst" is "Kvadratmeterpris" in "Grundværdispecifikation"
            for grundværdi_item in item.get("Grundværdispecifikation", []):
                if grundværdi_item["tekst"].strip() == "Kvadratmeterpris":
                    extracted_item["grund_kvadratmeterpris_beløb_justeret"] = grundværdi_item["beløb"]
                    break  # Stop the loop once found
            
            extracted_data.append(extracted_item)

    return extracted_data

#? Data retrieving functions
async def get_adresse_from_erb(BFE_number, session):
    data = await my_functions.get_EBR_data_from_BFE(BFE_number, session=session)
    features = data.get("features")
    assert len(features) == 1
    features = features[0].get("properties")
    Ejendomstype = features.get("Ejendomstype")
    husnummer = features.get("husnummer")
    assert len(husnummer) == 1
    adresse = husnummer[0].get("adgangsadressebetegnelse")
    kommune = husnummer[0].get("kommuneinddeling").get("navn")
    return (adresse, kommune, Ejendomstype)

async def get_BBR_grund_data_from_BFE(BFENUMMER, session):
    BBR_grund_data = await my_functions.get_BBR_grund_data_from_BFE(BFENUMMER, session)
    grund_id_lst = []
    for item_no in range(len(BBR_grund_data)):
        grund_id_lst.append(BBR_grund_data[item_no].get("id_lokalId"))
    return grund_id_lst



BygAnvendelse_kode_list = ['120', '121', '122', '131', '132', '190', '540']
BygAnvendelse_kode_text = ["Fritliggende enfamiliehus", "Sammenbygget enfamiliehus", "Fritliggende enfamiliehus i tæt-lav bebyggelse", 
                    "Række-, kæde- og klyngehus", "Dobbelthus", "Anden bygning til helårsbeboelse","Kolonihavehus"]

async def get_BBR_bygning_data_from_grund_id(grund_id, bfe, session):
    BBR_bygning_data = await my_functions.get_BBR_bygning_data_from_grund_id(grund_id, session)
    relevant_items = [item for item in BBR_bygning_data if item.get('byg021BygningensAnvendelse') in BygAnvendelse_kode_list]
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

    # define the list to return and get attributes
    list_to_return = []
    for bygningsnummer, items in grouped.items():
        sorted_items = sorted(items, key=lambda x: datetime.fromisoformat(x['virkningFra']), reverse=True)
        
        newest_item = sorted_items[0]
        anvendelse_kode_text = BygAnvendelse_kode_text[BygAnvendelse_kode_list.index(newest_item.get("byg021BygningensAnvendelse"))]
        
        # Add attributes of the newest item as tuple elements
        newest_item_tuple = (
            newest_item.get("id_lokalId"),
            newest_item.get("byg007Bygningsnummer"),
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
                'byg007Bygningsnummer': older_item.get("byg007Bygningsnummer"),
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

async def get_BBR_enhed_data_from_bygning_id(bygning_id, session):
    BBR_enhed_data = await my_functions.get_BBR_enhed_data_from_bygning_id(bygning_id, session)
    BBR_enhed_item = [item for item in BBR_enhed_data if item.get("enh023Boligtype") == "1"]
    assert len(BBR_enhed_item) == 1
    OBS_data = [item for item in BBR_enhed_data if item.get("enh023Boligtype") != "1"]
    return (BBR_enhed_item[0].get("enh031AntalVærelser"), 
            BBR_enhed_item[0].get("enh026EnhedensSamledeAreal"),
            BBR_enhed_item[0].get("enh027ArealTilBeboelse"),
            BBR_enhed_item[0].get("enh065AntalVandskylledeToiletter"),
            BBR_enhed_item[0].get("enh066AntalBadeværelser"),
            OBS_data)

async def get_vur_data(bfe_id, session):
    vurdata = await my_functions.get_vur_by_BFE(bfe_id, session)
    return filter_vur_data(vurdata)

async def get_property_size(bfe_id, session):
    return extract_attribute(await my_functions.get_MAT_from_BFE(bfe_id, session), "registreretAreal")

async def get_data_from_adress_refactored(bfeCode, session):
    step = 0
    try:
        flag = "Off"
        # Ejendomsbeliggenhedsregistret (EBR)
        try:
            adresse, kommune, Ejendomstype = await get_adresse_from_erb(bfeCode, session)
            item_data = {"adresse": adresse}
            item_data['Kommune'] = kommune
            item_data['Ejendomstype'] = Ejendomstype
        except:
            raise Exception("flag")
        item_data['BFE_nummer'] = bfeCode
        step=1

        # Bygnings- og Boligregistret (BBR): Grund
        grund_ids = await get_BBR_grund_data_from_BFE(bfeCode, session)
        step=2
        # Bygnings- og Boligregistret (BBR): Bygning
        item_data["BBR_bygning_data"] = []
        field_names = ['Bygningsnummer', 'BygningensSamledeBoligAreal', 'BebyggetAreal', 'ArealIndbyggetGarage', 'Opførelsesår', 'OmTilbygningsår', 'BygningensAnvendelse', 'byg404Koordinat', 'byg406Koordinatsystem', 'gamleBygning']
        bygning_ids_lst = []
        if grund_ids:
            for grund_id in grund_ids:
                # print("GRUND", grund_id, "of", len(grund_ids))
                bygning_data = {}
                list_returned = await get_BBR_bygning_data_from_grund_id(grund_id, bfeCode, session)
                try:
                    for item in list_returned:
                        try:
                            bygning_id = item[0]
                            bygning_ids_lst.append(bygning_id)
                            bygning_data = dict(zip(field_names, item[1:]))
                            item_data['BBR_bygning_data'].append(bygning_data)
                        except:
                            pass
                except Exception as e:
                    # print(f"Exception raised in 195: {e}")
                    pass
        
        step=3
        # Bygnings- og Boligregistret (BBR): Enhed
        item_data['BBR_enhed_data'] = []
        try:
            field_names = ['AntalVærelser', 'EnhedensSamledeAreal', 'ArealTilBeboelse', 'AntalVandskylledeToiletter', 'AntalBadeværelser', 'OBS']
            if bygning_ids_lst:
                for bygning_id in bygning_ids_lst:
                    bygning_data = {}
                    try:
                        values = await get_BBR_enhed_data_from_bygning_id(bygning_id, session)
                        bygning_data = dict(zip(field_names, values))
                    except Exception as e:
                        # print(f"l211 error: {e}")
                        bygning_data = {field: 'None' for field in field_names}
                    
                    item_data['BBR_enhed_data'].append(bygning_data)
            else:
                # print("bygning_ids_lst empty: Flag raised")
                raise Exception("flag")
        except Exception as e:
            if str(e) != "flag":
                print("FAILED 219")
            raise Exception(e)
        step=4

        # Get property assesments
        item_data['VUR_data'] = []
        vur_data_item_assesments = await get_vur_data(bfeCode, session)
        # verify lengths of boliger above
        sorted_VUR_data = sorted(vur_data_item_assesments, key=lambda x: x.get("ændringDato", ""), reverse=True)
        most_recent_entry = sorted_VUR_data[0]
        most_recent_antalMedvirk = most_recent_entry.get("antalMedvirk", None)
        item_data['VUR_data'] = vur_data_item_assesments
        
        if most_recent_antalMedvirk != None and most_recent_antalMedvirk != 0:
            if int(most_recent_antalMedvirk) != len(bygning_ids_lst):   
                my_base_functions.track_in_local_list("investigate.json", bfeCode)
                raise Exception("flag")
        
        step=5
        try:
            item_data['registreretAreal'] = await get_property_size(bfeCode, session)
        except:
            item_data['registreretAreal'] = None
        
        if flag == "Off":
            return item_data
        else:
            return flag
    except Exception as e:
        if str(e) == "flag":
            return "On"
        else:
            raise Exception(f"Failed in step {step}: {bfeCode} with {e} : 242")

### ! Async data fetcher

async def save_cache():
    global cache_count, cache_count_failed, cache_list, cached_ids, cache_list_failed, cached_ids_failed
    print("Saving cache...")
    with open("./data_cache.json", "w") as fp:
        while not response_queue.empty():
            item = await response_queue.get()
            cache_list.append(item)
            cache_count += 1
        json.dump(cache_list, fp)
    
    with open("./data_cache_failed.json", "w") as fp:
        while not failed_queue.empty():
            item = await failed_queue.get()
            cache_list_failed.append(item)
            cache_count_failed += 1
        jsonelem = {'BFE_nummer': cache_list_failed}
        json.dump(jsonelem, fp)
    print(f"Cache saved (S{len(cache_list)}:F{len(cache_list_failed)})")

    # update cache_ids
    cached_ids = [item['BFE_nummer'] for item in cache_list]
    cached_ids_failed = [item for item in cache_list_failed]

progress_lock = asyncio.Semaphore(1)

async def update_progress_bar_async():
    global pbar
    async with progress_lock:
        current_value = pbar.currval
        pbar.update(current_value + 1)

number_flagged_items = 0

async def fetch(semaphore, session, item_bfe, attempt):
    
    async with rate_limiter:
        try:
            item_json_response = await get_data_from_adress_refactored(item_bfe, session)
            if item_json_response != "On":
                await response_queue.put(item_json_response)
            else:
                global number_flagged_items
                number_flagged_items += 1
                await failed_queue.put(item_bfe)
            if response_queue.qsize() % GLOBAL_CACHED_COUNTER_SIZE == 0 and response_queue.qsize() > 0:
                await save_cache()
        except Exception as e:
            # if failed
            print(f"FAILED: {e}")
            await failed_queue.put(item_bfe)

        await update_progress_bar_async()
        
        if semaphore.locked(): # check if the semaphore limit has been reached
            semaphore.release() # increment the semaphore limit
    
async def fetch_with_sem(semaphore, session, item, attempt = 1):
    async with semaphore:
        result = await fetch(semaphore, session, item, attempt)
    return result

class CustomProgressBar(progressbar.ProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()

    def update(self, value=None):
        elapsed_time = time.time() - self.start_time
        speed = value / elapsed_time * 60 if elapsed_time > 0 else 0
        self.widgets[-1] = f'Speed: {speed:.2f} req/min'
        super().update(value)

async def fetch_all(session, semaphore):
    global tasks, pbar, total_queued

    # Initialize variables
    widgets = [
        'Progress: ', 
        progressbar.SimpleProgress(), 
        progressbar.Percentage(), 
        ' ',
        progressbar.Bar(marker='#', left='[', right=']'), 
        ' ',
        progressbar.ETA(),
        ' '
    ]

    tasks = []
    total_queued = 0
    total_notqueued = 0

    # Load and filter BFE numbers
    bfe_number_list = load_and_filter_BFE_numbers()

    print("Start queueing")

    # Create tasks
    for bfe_number in bfe_number_list:
        if bfe_number not in cached_ids and bfe_number not in cached_ids_failed:
            tasks.append(fetch_with_sem(semaphore, session, bfe_number))
            total_queued += 1
            # When we reach 200,000 tasks, gather and wait for them to complete
            if total_queued % GLOBAL_QUEUE_SIZE == 0:
                print(f"Gathering {total_queued} items")
                pbar = CustomProgressBar(widgets=widgets, maxval=total_queued)
                pbar.start()
                await asyncio.gather(*tasks, return_exceptions=True)
                tasks = []  # Reset tasks
                pbar.finish()

        else:
            total_notqueued += 1

    # Handle any remaining tasks that didn't make up a full batch of 200,000
    if tasks:
        print(f"Gathering remaining {len(tasks)} items")
        pbar = CustomProgressBar(widgets=widgets, maxval=len(tasks))
        pbar.start()
        await asyncio.gather(*tasks, return_exceptions=True)
        pbar.finish()

    print(f"Items queued: {total_queued}")
    print(f"Items not queued: {total_notqueued}")
    print(f"Number of completed tasks: {response_queue.qsize()}")

async def main():
    async with httpx.AsyncClient(timeout=None) as session:
        # create progress bar and timer
        semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
        # wait for all tasks to complete
        await fetch_all(session, semaphore)
        # write the results to a JSON file
        print("Exporting")
        #? Responce queue:
        await save_cache()
        print("Done")

if __name__ in '__main__':
    # Start request timer
    start = perf_counter()
    # Start requests
    asyncio.run(main())
    # End timer
    stop = perf_counter()
    print(f"It took {round(stop-start,2)} seconds to make {int(cache_count)} API calls ({round(cache_count/(stop-start),2)} calls pr. second)")
    print(f"{number_flagged_items} flagged of {total_queued}")