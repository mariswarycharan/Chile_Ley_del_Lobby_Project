import pandas as pd

# Load the Excel file
file_path = 'leylobby_doctors_data_filled(Institute of Public Health Translated).xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Group by Identifier to aggregate meeting details
grouped = df.groupby('Identifier', group_keys=False)

# Define a function to generate stories in the exact format provided
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
        details = f"{row['Assistant_Full_name']} ({row['Assistant_Quality']}, working for {row['Assistant_Works_for']}, representing {row['Assistant_Represents']})"
        assistant_details.append(details)
    assistant_text = "\n".join(assistant_details)


    # Compile story
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

# Generate formatted stories for all meetings
formatted_stories = grouped.apply(generate_formatted_story)

# Save the stories to a text file with UTF-8 encoding
output_file = 'enhanced_meeting_details_story(Institute of Public Health Translated).txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for story in formatted_stories:
        f.write(story + "\n\n---\n\n")

print(f"Formatted stories have been saved to {output_file}.")