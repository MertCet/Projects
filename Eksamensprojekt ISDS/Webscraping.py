
################################ Packages #################################
from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep, time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
import pandas as pd
from selenium.webdriver.common.action_chains import ActionChains
import os
import requests
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
import random


################################ START #################################

# browser settings
options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True) # Without this code, the chrome tab will close immediately
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument('log-level=3')

url = "https://home.dk/"

class home_bot():

    def initialize_driver(self):
        self.driver = webdriver.Chrome(service=Service(), options=options)
        self.driver.maximize_window()
        self.driver.get(url)
        global wait, DELAY
        DELAY = 3
        wait = WebDriverWait(self.driver, DELAY)

    # Accept cookies
    def accept_cookies(self):
        cookie_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'coi-banner__accept')))
        cookie_button.click()

    # Navigate to the list containing properties for sale
    def navigate_to_property_list(self):
        menu_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'handle-text')))
        menu_button.click()
        sog_bolig_button = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, 'Søg bolig')))
        sog_bolig_button.click()

    # Filter the list to only contain necessary elements
    def filter_properties(self):
        label_leje = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[contains(text(),'Boliger til leje*')]")))
        label_leje.click()

        #list of desired property types
        ejendomstype_click = ["Villa", "Rækkehus", "Lejlighed", "Landejendom", "Andelsbolig", "Villalejlighed"]
        
        #Loop through each labeltext and click on the corresponding label
        for ejendom in ejendomstype_click:
            ejendom_element = wait.until(EC.element_to_be_clickable((By.XPATH, f"//div[@id='estate-type']//label[contains(text(), '{ejendom}')]")))
            ejendom_element.click()

        sleep(2)
        WebDriverWait(self.driver, DELAY).until(EC.element_to_be_clickable((By.XPATH, "//div[@class='selectric']//p[@class='label' and text()='Sorter']"))).click()
        WebDriverWait(self.driver, DELAY).until(EC.element_to_be_clickable((By.XPATH, "//div[@class='selectric-items']//li[text()='Liggetid (ældste først )']"))).click()

    # Check if the "boliger" count has changed from "0 boliger"
    def wait_for_non_zero_boliger_count(self, timeout=60):  # Added timeout parameter
        start_time = time()
        while time() - start_time < timeout:
            try:
                boliger_element = wait.until(EC.presence_of_element_located((By.XPATH, "//h1[@class='xl-heading']")))
                boliger_count = boliger_element.text.split()[0]  # Grab the first part of "int boliger"
                if int(boliger_count) > 0:  # Check if the count is greater than zero
                    return True
            except Exception as e:
                print(f"An error occurred while waiting for non-zero boliger count: {e}")

            sleep(0.5)
        raise TimeoutError("Timeout while waiting for non-zero boliger count")


global bot
bot = home_bot()


################################ Postal Code Filter #################################


def hent_postnumre():
    # URL of the webpage you want to scrape
    url = "https://www.postnumre.dk/"

    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    soup = BeautifulSoup(response.content, 'lxml')

    # Extract all <tr> tags from the table
    rows = soup.find('table').find_all('tr')

    # Extract headers from the first row
    headers = [header.text.strip() for header in rows[0].find_all('td')]

    # Extract data from the subsequent rows
    result = []
    for row in rows[1:]:
        data = {}
        for header, td in zip(headers, row.find_all('td')):
            data[header] = td.text.strip()
        result.append(data)
    global postal_codes
    postal_codes = [entry["Postnr."] for entry in result if int(entry["Postnr."]) >= 6400]




################################ SCRAPE DATA #################################



def scroll_to_bottom():
    result_gallery  = WebDriverWait(bot.driver, DELAY).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="result-gallery"]')))
    infinite_scroller_div = result_gallery.find_element(By.XPATH, ".//div[@infinite-scroller]")
    previous_count = 0
    print("NEW PAGE: Starting at 0")
    while True:
        actions.move_to_element(infinite_scroller_div).send_keys(Keys.END).perform()
        try:
            WebDriverWait(bot.driver, 10).until(
                lambda x: len(infinite_scroller_div.find_elements(By.XPATH, ".//section[@itemprop='itemListElement']")) > previous_count
            )
            sleep(2)
        except:
            try: 
                WebDriverWait(bot.driver, 10).until(
                    lambda x: len(infinite_scroller_div.find_elements(By.XPATH, ".//section[@itemprop='itemListElement']")) > previous_count
                )
                sleep(2)
            except:
                break
        previous_count = len(infinite_scroller_div.find_elements(By.XPATH, ".//section[@itemprop='itemListElement']"))
        print(previous_count)




def main_function():
    
    filename = "home_data.csv"
    start_url = "https://home.dk/resultatliste/?IsBusinessSearch=false&SortType=Time&SortOrder=desc&CurrentPageNumber=0&SearchResultsPerPage=15&EjendomstypeV1=true&EjendomstypeRH=true&EjendomstypeEL=true&EjendomstypeVL=true&EjendomstypeAA=true&EjendomstypeLO=true&Energimaerker=null&BoligKanLejes=false&BoligTilSalg=true&SearchType=0"

    # Iterate over the postal codes
    for postal_code in postal_codes:
        
        
        # Click on the search input field (if this is necessary)
        search_field = WebDriverWait(bot.driver, DELAY).until(
            EC.element_to_be_clickable((By.XPATH, '//button[@data-form="#search" and @class="toggle-item-new handle-search-icon js-handle-search js-trigger-form"]'))
        )
        search_field.click()
        sleep(2)  # Adding a delay to ensure any overlay or modal triggered by the click has time to show up
        
        # Find the search input field and enter the postal code
        search_input = WebDriverWait(bot.driver, DELAY).until(
            EC.element_to_be_clickable((By.XPATH, '//input[@id="query" and @data-is-overlay="true"]'))
        )
        search_input.clear()  # Clear any previous input
        search_input.send_keys(postal_code)
        sleep(2)
        # Click on the search link
        search_link = WebDriverWait(bot.driver, DELAY).until(
            EC.element_to_be_clickable((By.XPATH, "//a[@class='search-link' and @ng-click='selectMatch($index)']"))
        )
        search_link.click()
        # Scroll to the top of the page
        bot.driver.execute_script("window.scrollTo(0, 0);")
        
        sleep(2)  # Allow some time for the results to load

        
        # Scroll to load more content
        scroll_to_bottom()
        
        # Your scraping code starts here
        result_gallery  = WebDriverWait(bot.driver, DELAY).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="result-gallery"]')))
        infinite_scroller_div = result_gallery.find_element(By.XPATH, ".//div[@infinite-scroller]")

        sleep(2)

        Address = []
        Postal_Code = []
        City = []
        Price = []
        Type = []

        previous_items_count = 0

        while True:  # Infinite loop, will break when no new items are loaded
            items = infinite_scroller_div.find_elements(By.XPATH, './section')

            # If no new items are loaded, break the loop
            if len(items) == previous_items_count:
                break

            for item in items:
                try:
                    price_element_text = item.find_element(By.XPATH, ".//span[@itemprop='price']").text
                except NoSuchElementException:  # This exception is raised when the element is not found
                    continue  # skip the current iteration and move to the next item

                if not price_element_text:  # skip if price is empty
                    continue

                address_element = item.find_element(By.XPATH, ".//span[@itemprop='address']/b[@itemprop='streetAddress']")
                
                # Append data to lists
                Address.append(address_element.text)
                Postal_Code.append(item.find_element(By.XPATH, ".//span[@itemprop='postalCode']").text)
                City.append(item.find_element(By.XPATH, ".//span[@itemprop='addressLocality']").text)
                Price.append(price_element_text)
                Type.append(item.find_element(By.XPATH, ".//span[@ng-if='result.ejendomstypePrimaerNicename']").text)



            # Save the data
            home_df = pd.DataFrame({
                'Address'     : Address,
                'Postal Code' : Postal_Code,
                'City'        : City,
                'Price'       : Price,
                'Type'        : Type
            })

            if os.path.exists(filename):
                home_df.to_csv(filename, mode='a', header=False, index=False, encoding='utf-8-sig', sep=';')
            else:
                home_df.to_csv(filename, mode='w', header=True, index=False, encoding='utf-8-sig', sep=';')
                    
            # Clear the lists for the next loop
            Address.clear()
            Postal_Code.clear()
            City.clear()
            Price.clear()
            Type.clear()

            # Store the current item count
            previous_items_count = len(items)
            
        bot.driver.get(start_url)
        sleep(random.uniform(2, 4)) 
        

if __name__ in '__main__':
    bot.initialize_driver()
    actions = ActionChains(bot.driver)
    bot.accept_cookies()
    bot.navigate_to_property_list()
    bot.wait_for_non_zero_boliger_count()  # Wait here until the count is non-zero
    bot.filter_properties()
    hent_postnumre()
    main_function()
    bot.driver.quit()





