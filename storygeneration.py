import pandas as pd
from tqdm import tqdm

def process_and_filter_data(input_file_path):

    df = pd.read_excel(input_file_path)

    columns_to_fill = ['full_name', 'Position', 'link', 'Identifier', 'date', 'Shape', 'Place',
    'Duration', 'Subjects_covered', 'Specification']

    # Fill missing values in specified columns
    for column in columns_to_fill:
        if column in df.columns:
            df[column] = df[column].fillna(method='ffill')
        else:
            print(f"Column '{column}' not found in the DataFrame.")

    
    company_names = [
    "ABBOTT LABORATORIES DE CHILE LTDA", "Abbott Laboratories of Chile",
    "ABBOTT LABORATORIES OF CHILE LDTA", "Abbvie", "Abbvie Laboratory Ltda",
    "Abbvie Ltda", "Abbvie Pharmaceutical Laboratory LTDA", "Abbvie Pharmaceuticals Ltd.",
    "ARQUIMED LIMITED", "ARQUIMED LTDA", "ARQUIMED LTDA.", "Astellas Farma Chile SpA",
    "Astellas Pharma Chile", "AstraZeneca Chile", "ASTRAZENECA S.A.", "Astrazeneca SA",
    "Biogen", "Biogen Chile SPA", "Biogen Chule SPA", "Biogen SPA Chile",
    "BIOTOSCANA FARMA SpA LABORATORY", "BMS", "Boehringer Ingelheim Ltd.",
    "BOEHRINGER INGELHEIM LTDA", "Boehringer Ingelheim Ltda.", "Bristol Myers",
    "Bristol Myers Squibb", "BRISTOL MYERS SQUIBB CHILE", "Bristol Myers Squibb Company",
    "CARE Oncology Foundation", "GLAXOSMITHKLINE CHILE FARMACEUTICA LIMITADA",
    "GlaxoSmithKline Chile Farmacéutica Ltda", "GlaxoSmithkline Chile Pharmaceuticals Limited",
    "GlaxoSmithKline Farmaceutica LTDA.", "GSK", "INNOVATIVE MEDICINES SA AGENCY IN CHILE",
    "IQMED CHILE SPA", "J&J SpA Marketing Company", "MEDICIP HEALTH SL",
    "MEDICIP HEALTH, SL", "MEDIPLEX", "MEDIPLEX SA", "Merck Laboratory SA",
    "MERCK SA Laboratory", "Merck Sharp & Dhome", "MERCK SHARP & DOHME",
    "Merck Sharp & Dohme (ia) Llc, agency in Chile", "MERCK SHARP & DOHME (IA) LLC. Agency in Chile",
    "National Cancer Corporation / CONAC", "National Center for Health Information Systems – CENS",
    "Novartis Chile", "Novartis Chile S.A.", "NOVARTIS S.A.", "Novo Nordisk Pharmaceuticals",
    "Pfizer", "Pfizer Chile S.A.", "Recemed", "Recemed SPA", "Roche Chile",
    "Roche Chile Limited", "Roche Chile Ltda", "Roche Chile Ltda.", "Rochem Biocare Chile Spa",
    "Sandoz Chile SPA", "Sanofi Aventis de Chile S.A.", "Sanofi Pasteur S.A.",
    "Sinovac Biotech (Chile) SpA", "Takeda Chile spa", "Vertex Pharmaceuticals",
    "Vertex Pharmaceuticals Inc.", "VITROSCIENCE SPA", "Biogen Chile SpA",
    "Boehringer Ingelheim Ltda", "Bristo Myers Squibb", "Bristol Myeres Squibb",
    "MERCK SA Laboratory", "Tecnofarma Chile S.A."
    ]

    # Filter the data for meetings involving specified companies
    matching_meetings = df[df['Assistant_Represents'].isin(company_names)]
    filtered_data = df[df['Identifier'].isin(matching_meetings['Identifier'])]

    return filtered_data

# Load the Excel file
file_path = 'output/leylobby_doctors_data_structured_translated_v2.xlsx'
df = process_and_filter_data(file_path)

# Group by Identifier to handle multi-entry meetings
grouped = df.groupby(['Identifier', 'date'])
# Function to generate formatted narrative story with summary and clear sections


def create_story(group):
    row = group.iloc[0]
    # Date and Time Formatting with Time of Day
    formatted_date = pd.to_datetime(row['date'], errors='coerce').strftime('%B %d, %Y') if pd.notna(
        row['date']) else 'an unspecified date'
    time = pd.to_datetime(row['date'], errors='coerce').strftime('%I:%M %p') if pd.notna(
        row['date']) else 'an unspecified time'
    hour = pd.to_datetime(row['date'], errors='coerce').hour if pd.notna(row['date']) else 12
    # Determine time of day
    if hour < 12:
        time_of_day = "morning"
    elif 12 <= hour < 18:
        time_of_day = "afternoon"
    else:
        time_of_day = "evening"
    duration = row.get('Duration', 'an unspecified duration')
    place = row.get('Place', 'an unspecified location')
    shape = row.get('Shape', 'a standard format')
    focus = row.get('Subjects_covered', 'general topics')
    identifier = row.get('Identifier', 'Unknown Meeting ID')
    leader = row.get('full_name', 'an official')
    position = row.get('Position', 'an unspecified role')
    link = row.get('link', '#')
    # Short summary for quick overview
    summary = (
        f"This meeting, led by **{leader}** on **{formatted_date}**, "
        f"focused on **{focus.lower()}** and lasted for **{duration}**."
    )
    # Start the narrative
    story = f"""
🆔 **Meeting Identifier**  
**{identifier}**  
📋 **Meeting Summary**  
{summary}  
📅 **Date and Time**  
On the {time_of_day} of **{formatted_date} at {time}**, a significant meeting was led by **{leader}**,  
serving as **{position}**.  
👤 **Meeting Leader**  
**{leader}** plays a critical role in addressing the **{focus.lower()}** as part of their responsibilities within the government framework.  
📍 **Location and Format**  
- **Location**: {place}  
- **Format**: {shape}  
- **Duration**: {duration}  
🎯 **Focus and Objectives**  
The primary focus of the discussion revolved around the **{focus}**.  
👥 **Participants and Representatives**  
Attendees actively contributed to the session, sharing insights and perspectives on the matter at hand.  
"""
    # Add participants in bullet point format
    participants = [
        f"- **{p.get('Assistant_Full_name', 'Unnamed')}** – {p.get('Assistant_Quality', 'Unknown')}, representing {p.get('Assistant_Works_for', 'N/A')}"
        for _, p in group.iterrows()
    ]
    if participants:
        story += "\n".join(participants)
    else:
        story += "- No participants recorded."
    # Add meeting notes and highlights
    notes = row.get('Specification', 'No additional notes were provided during the session.')
    story += f"""
📄 **Meeting Highlights and Notes**  
During the meeting, the following topics and points were discussed:  
- {notes}  
🔗 **Official Record Link**  
For further details, [access the full meeting record here]({link}).  
📊 **Summary and Significance**  
The meeting lasted **{duration}** and concluded with actionable insights regarding {focus.lower()}.  
The contributions of the attendees enriched the discussion, fostering transparency and comprehensive decision-making.  
This meeting highlights the **dedication and collaborative efforts** of public officials and stakeholders in shaping key policies and programs for the betterment of governance and public service.  
==============================================================================================================================
"""
    return story

# Generate full narrative story for each meeting
full_story = "".join( create_story(group) for _, group in tqdm(grouped))

# Save the output to a text file
output_path = 'output/meeting_details_story.txt'
with open(output_path, 'w', encoding='utf-8') as file:
    file.write(full_story)
    
print(f"Narrative story saved to: {output_path}")
