import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import os
import random
import datetime
from deep_translator import GoogleTranslator  # Import deep translator
import webbrowser

# Toggle this when you actually want to open links
OPEN_DETAIL_LINKS = False   # opens each meeting detail link
OPEN_LISTING_PAGES = True  # opens the paginated listing pages


# Define a helper function to handle requests with retries
def safe_get(url, retries=3, delay=3):
    for i in range(retries):
        try:
            response = requests.get(url.strip())
            response.raise_for_status()  # Raise an error for bad HTTP status codes
            return response
        except requests.exceptions.RequestException as e:
            time.sleep(delay * (i + 1))  # Exponential backoff
    return None  # Return None if retries are exhausted


# Function to translate text using deep translator
#def translate_text(text, target_language='en'):
#    if not text or pd.isna(text):
#        return text
#    try:
#        return GoogleTranslator(source='auto', target=target_language).translate(text)
#    except Exception as e:
#        # In case of any translation issues, return the original text
#        return text


# Function to extract data from the first link
def first_link_data(link):
    doctor_info = {}
    response = safe_get(link)
    if not response:
        return doctor_info  # Return empty data if the request fails
    soup = BeautifulSoup(response.text, 'html.parser')
    table_data = soup.find('div', class_='col-12')
    if table_data:
        first_table_data = table_data.find('table')
        if first_table_data:
            rows = first_table_data.find_all('tr')
            doctor_info['Identifier'] = rows[0].find_all('td')[1].get_text(strip=True) if len(rows) > 0 else ''
            doctor_info['date'] = rows[1].find_all('td')[1].get_text(strip=True) if len(rows) > 1 else ''
            doctor_info['Shape'] = rows[2].find_all('td')[1].get_text(strip=True) if len(rows) > 2 else ''
            doctor_info['Place'] = rows[3].find_all('td')[1].get_text(strip=True) if len(rows) > 3 else ''
            doctor_info['Duration'] = rows[4].find_all('td')[1].get_text(strip=True) if len(rows) > 4 else ''
    table = soup.find_all('tbody')
    if table:
        list_for_table1 = []
        table1_rows = table[0].find_all('tr')
        for row in table1_rows:
            columns = row.find_all('td')
            if len(columns) == 4:
                list_for_table1.append({
                    'Full_name': columns[0].get_text(strip=True),
                    'Quality': columns[1].get_text(strip=True),
                    'Works_for': columns[2].get_text(strip=True),
                    'Represents': columns[3].get_text(strip=True)
                })
        doctor_info['Assistants'] = list_for_table1
        doctor_info['Subjects_covered'] = ", \n".join(
            [i.find('td').get_text(strip=True) for i in table[1].find_all('tr')]) if len(table) > 1 else ''
        doctor_info['Specification'] = table[2].find('tr').find('td').get_text(strip=True) if len(table) > 2 else ''
    return doctor_info


# Function to structure data (expand assistant info into separate rows)
def structuring_data(df):
    assistants_column = 'Assistants'
    expanded_rows = []
    if assistants_column in df.columns:
        for _, row in df.iterrows():
            assistants_data = row[assistants_column]
            assistants_list = assistants_data
            is_first_row = True
            for assistant in assistants_list:
                new_row = row.copy()
                new_row['Assistant_Full_name'] = assistant.get('Full_name', None)
                new_row['Assistant_Quality'] = assistant.get('Quality', None)
                new_row['Assistant_Works_for'] = assistant.get('Works_for', None)
                new_row['Assistant_Represents'] = assistant.get('Represents', None)
                if not is_first_row:
                    for col in row.index:
                        if col not in ['Assistant_Full_name', 'Assistant_Quality', 'Assistant_Works_for',
                                       'Assistant_Represents']:
                            new_row[col] = None
                else:
                    is_first_row = False
                expanded_rows.append(new_row)
            if not assistants_list:
                new_row = row.copy()
                new_row['Assistant_Full_name'] = None
                new_row['Assistant_Quality'] = None
                new_row['Assistant_Works_for'] = None
                new_row['Assistant_Represents'] = None
                expanded_rows.append(new_row)
    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df = expanded_df[df.columns.tolist() + ['Assistant_Full_name', 'Assistant_Quality', 'Assistant_Works_for',
                                                     'Assistant_Represents']]
    expanded_df.drop(columns=[assistants_column], inplace=True)
    return expanded_df


def scrape_data():
    current_year = datetime.datetime.now().year  # Use current year dynamically
    institutions = {
        'AO001': 'Ministry of Health',
        'AO003': 'Supply Center of the National Health Services System',
        'AO004': 'National Health Fund',
        'AO005': 'Institute of Public Health',
        'AO006': 'Superintendency of Health'
    }
    years = [current_year]  # Only scrape for the current year
    entire_doctors_list = []
    # List to collect file names of saved Excel files.
    saved_excel_files = []

    for institution, institution_name in institutions.items():
        for year in years:
            entire_doctors_list = []
            for i in range(1, 3):
                # Iterate through pages 1 and 2
                page_url = f'https://www.leylobby.gob.cl/instituciones/{institution}/audiencias/{year}?page={i}'
                if OPEN_LISTING_PAGES:
                    webbrowser.open_new_tab(page_url)
                response = safe_get(page_url)
                if not response:
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                tbody = soup.find('tbody')
                rows = tbody.find_all('tr') if tbody else []

                for row in tqdm(rows, desc=f"Processing {institution} {year}, page {i}"):
                    columns = row.find_all('td')
                    if columns:
                        full_name = columns[0].get_text(strip=True)
                        Position = columns[1].get_text(strip=True)
                        link = columns[2].find('a')['href']
                        response = safe_get(link)
                        if not response:
                            continue
                        soup = BeautifulSoup(response.text, 'html.parser')
                        tbody = soup.find('tbody')
                        sub_rows = tbody.find_all('tr') if tbody else []
                        for sub_row in tqdm(sub_rows, desc="Processing subrows"):
                            columns = sub_row.find_all('td')
                            if len(columns) == 7:
                                link_data = columns[6].find('a')['href']
                                if OPEN_DETAIL_LINKS:
                                    webbrowser.open_new_tab(link_data)
                                time.sleep(random.uniform(1, 2))
                                first_link_data_dist = first_link_data(link_data)
                                entire_doctors_list.append({
                                    'full_name': full_name,
                                    'Position': Position,
                                    'link': link_data,
                                    'Identifier': first_link_data_dist.get('Identifier', ''),
                                    'date': first_link_data_dist.get('date', ''),
                                    'Shape': first_link_data_dist.get('Shape', ''),
                                    'Place': first_link_data_dist.get('Place', ''),
                                    'Duration': first_link_data_dist.get('Duration', ''),
                                    'Assistants': first_link_data_dist.get('Assistants', []),
                                    'Subjects_covered': first_link_data_dist.get('Subjects_covered', ''),
                                    'Specification': first_link_data_dist.get('Specification', '')
                                })

            df = pd.DataFrame(entire_doctors_list)

            # Translate specified columns before further processing
#            translation_columns = ['Position', 'Shape', 'Place', 'Subjects_covered', 'Specification']
#            for col in translation_columns:
#                if col in df.columns:
#                    df[col] = df[col].apply(lambda x: translate_text(x, target_language='en') if x else x)

            # For the Assistants column, translate each inner field
#            if 'Assistants' in df.columns:
#                def translate_assistants(assistants_list):
#                    if not assistants_list:
#                        return assistants_list
#                    for assistant in assistants_list:
#                       assistant['Full_name'] = translate_text(assistant.get('Full_name', ''), target_language='en')
#                        assistant['Quality'] = translate_text(assistant.get('Quality', ''), target_language='en')
#                        assistant['Works_for'] = translate_text(assistant.get('Works_for', ''), target_language='en')
#                        assistant['Represents'] = translate_text(assistant.get('Represents', ''), target_language='en')
#                    return assistants_list

#                df['Assistants'] = df['Assistants'].apply(translate_assistants)

            # Structure the data so that each assistant gets its own row
            df = structuring_data(df)

            # Optionally, translate the assistant columns in the structured dataframe as well
 #           assistant_columns = ['Assistant_Full_name', 'Assistant_Quality', 'Assistant_Works_for',
  #                               'Assistant_Represents']
 #           for col in assistant_columns:
 #               if col in df.columns:
 #                   df[col] = df[col].apply(lambda x: translate_text(x, target_language='en') if x else x)

#            if not os.path.exists(f'output/{str(year)}'):
#                os.makedirs(f'output/{str(year)}')
#
            if not os.path.exists(f'output/{str(year)}/scraped_data'):
                os.makedirs(f'output/{str(year)}/scraped_data')

            file_name_structured = f'output/{str(year)}/scraped_data/{institution_name} {year}.xlsx'
            df.to_excel(file_name_structured, index=False)

            # Add file name to saved_excel_files list
            #saved_excel_files.append(file_name_structured)

    # Save the list of generated Excel file names to a text file for further usage.
    #with open('output/excel_filenames.txt', 'w') as f:
        #for file in saved_excel_files:
            #f.write(file + "\n")

# To run the scraper, simply call the function:
#scrape_data()
