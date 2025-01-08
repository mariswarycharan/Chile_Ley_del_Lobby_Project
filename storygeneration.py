import pandas as pd
# Load the Excel file
file_path = 'leylobby_doctors_data_filled.xlsx'
df = pd.read_excel(file_path)
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
full_story = "".join(create_story(group) for _, group in grouped)
# Save the output to a text file
output_path = 'output/enhanced_meeting_details_story.txt'
with open(output_path, 'w', encoding='utf-8') as file:
    file.write(full_story)
print(f"Narrative story saved to: {output_path}")
