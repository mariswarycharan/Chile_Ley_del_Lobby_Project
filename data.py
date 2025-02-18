import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import random
import datetime

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

# Function to extract data from the first link
def first_link_data(link):
    doctor_info = {}
    response = safe_get(link)
    if not response:
        return doctor_info  # Return empty data if the request fails
    soup = BeautifulSoup(response.text, 'html.parser')
    table_data = soup.find('div', class_='col-xs-6')
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
        doctor_info['Subjects_covered'] = ", \n".join([i.find('td').get_text(strip=True) for i in table[1].find_all('tr')]) if len(table) > 1 else ''
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
                        if col not in ['Assistant_Full_name', 'Assistant_Quality', 'Assistant_Works_for', 'Assistant_Represents']:
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
    expanded_df = expanded_df[df.columns.tolist() + ['Assistant_Full_name', 'Assistant_Quality', 'Assistant_Works_for', 'Assistant_Represents']]
    expanded_df.drop(columns=[assistants_column], inplace=True)
    return expanded_df

def fill_missing_values(file_path, output_path):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    if 'Identifier' in df.columns:
        df['Identifier'] = df['Identifier'].astype(str).str.strip()
    df_filled = df.copy()
    assistant_columns = [col for col in df.columns if col.startswith('Assistant_')]
    columns_to_fill = [col for col in df.columns if not col.startswith('Assistant_')]
    for identifier in df['Identifier'].dropna().unique():
        mask = df['Identifier'] == identifier
        if mask.sum() > 1:
            group_rows = df_filled[mask]
            first_row_with_data = group_rows.iloc[0]
            for column in columns_to_fill:
                for idx in group_rows.index:
                    if pd.isnull(df_filled.at[idx, column]) and not pd.isnull(first_row_with_data[column]):
                        df_filled.at[idx, column] = first_row_with_data[column]
    rows_to_process = df_filled.index
    for idx in rows_to_process:
        current_row = df_filled.iloc[idx]
        if not pd.isnull(current_row['Assistant_Full_name']) and current_row[columns_to_fill].isnull().any():
            search_range = 3
            for i in range(1, search_range + 1):
                if idx - i >= 0:
                    prev_row = df_filled.iloc[idx - i]
                    if not prev_row[columns_to_fill].isnull().all():
                        for column in columns_to_fill:
                            if pd.isnull(df_filled.at[idx, column]) and not pd.isnull(prev_row[column]):
                                df_filled.at[idx, column] = prev_row[column]
                        if pd.isnull(df_filled.at[idx, 'Identifier']) or df_filled.at[idx, 'Identifier'] != prev_row['Identifier']:
                            df_filled.at[idx, 'Identifier'] = prev_row['Identifier']
                        break
            if df_filled.iloc[idx][columns_to_fill].isnull().any():
                for i in range(1, search_range + 1):
                    if idx + i < len(df_filled):
                        next_row = df_filled.iloc[idx + i]
                        if not next_row[columns_to_fill].isnull().all():
                            for column in columns_to_fill:
                                if pd.isnull(df_filled.at[idx, column]) and not pd.isnull(next_row[column]):
                                    df_filled.at[idx, column] = next_row[column]
                            if pd.isnull(df_filled.at[idx, 'Identifier']) or df_filled.at[idx, 'Identifier'] != next_row['Identifier']:
                                df_filled.at[idx, 'Identifier'] = next_row['Identifier']
                            break
    for column in columns_to_fill:
        df_filled[column] = df_filled[column].ffill()
        df_filled[column] = df_filled[column].bfill()
    if 'Identifier' in df_filled.columns:
        df_filled['Identifier'] = df_filled['Identifier'].ffill()
        df_filled['Identifier'] = df_filled['Identifier'].bfill()
    df_filled.to_excel(output_path, index=False, engine='openpyxl')

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
            for i in range(1, 3):  # Iterate through pages 1 and 2
                page_url = f'https://www.leylobby.gob.cl/instituciones/{institution}/audiencias/{year}?page={i}'
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
            file_name = f'output/{institution_name} {year}.xlsx'
            df.to_excel(file_name, index=False)
            saved_excel_files.append(file_name)
            df = structuring_data(df)
            file_name_structured = f'output/{institution_name} {year} structured.xlsx'
            df.to_excel(file_name_structured, index=False)
            fill_missing_values(file_name_structured, f'output/{institution_name} {year}.xlsx')

    # Save the list of generated Excel file names to a text file for narrative.py usage.
    with open('output/excel_filenames.txt', 'w') as f:
        for file in saved_excel_files:
            f.write(file + "\n")