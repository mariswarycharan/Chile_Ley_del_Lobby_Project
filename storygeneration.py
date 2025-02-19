import os
import pandas as pd
import datetime

def fill_missing_values(df):
    
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
    
    return df_filled

def generate_formatted_story(group):
    # Extract general meeting details (assume they are the same within a group)
    official_name = group['full_name'].iloc[0]
    position = group['Position'].iloc[0]
    date = group['date'].iloc[0]
    shape = group['Shape'].iloc[0]
    place = group['Place'].iloc[0]
    duration = group['Duration'].iloc[0]
    subjects_covered = group['Subjects_covered'].iloc[0]
    specification = group['Specification'].iloc[0]

    # Extract assistants' details with all attributes
    assistants = group[['Assistant_Full_name', 'Assistant_Quality', 'Assistant_Works_for', 'Assistant_Represents']]
    assistant_details = []
    for _, row in assistants.iterrows():
        details = (f"{row['Assistant_Full_name']} "
                   f"({row['Assistant_Quality']}, working for {row['Assistant_Works_for']}, "
                   f"representing {row['Assistant_Represents']})")
        assistant_details.append(details)
    assistant_text = "\n".join(assistant_details)

    # Compile the formatted story
    story = (
        f"On {pd.to_datetime(date).strftime('%B %d, %Y')} at {pd.to_datetime(date).strftime('%I:%M %p')}, "
        f"{official_name}, the {position}, attended a {shape.lower()} meeting via {place}. "
        f"The meeting lasted {duration} and focused on the {subjects_covered}.\n\n"
        f"**Purpose:**\n{specification}.\n\n"
        f"**Participants:**\nThe meeting included:\n{assistant_text}\n\n"
        f"**Key Details:**\n"
        f"Meeting Identifier: {group['Identifier'].iloc[0]}\n"
        f"Platform: {place}\n"
        f"Duration: {duration}\n"
        f"Subjects Discussed: {subjects_covered}\n"
    )
    return story


def generate_narratives():
    current_year = str(datetime.datetime.now().year)

    # Use f-string for proper interpolation
    output_dir = f"output/{current_year}/story_files"

    if not os.path.exists(f'output/{current_year}'):
        os.makedirs(f'output/{current_year}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    excel_files = [f for f in os.listdir(f'output/{current_year}/scraped_data')
                   if f.endswith(".xlsx") and current_year in f]

    if not excel_files:
        print("No Excel files for the current year found in the output directory.")
        return

    for file_name in excel_files:
        try:
            # Adjust this if necessary: ensure you're reading from the correct directory
            file_path = os.path.join("output", current_year, "scraped_data", file_name)
            df = pd.read_excel(file_path)

            df = fill_missing_values(df)

            grouped = df.groupby('Identifier', group_keys=False)
            formatted_stories = grouped.apply(generate_formatted_story)
            base_name = os.path.splitext(file_name)[0]
            output_file = os.path.join(output_dir, f"{base_name}.txt")

            with open(output_file, "w", encoding="utf-8") as f_out:
                for story in formatted_stories:
                    f_out.write(story + "\n\n---\n\n")
            print(f"Formatted stories have been saved to {output_file}.")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")
