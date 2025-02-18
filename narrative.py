import os
import pandas as pd
import datetime


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
    output_dir = "C:/Users/M.HARISH/PycharmProjects/Chile_Ley_del_Lobby_Project/output"

    current_year = str(datetime.datetime.now().year)

    excel_files = [f for f in os.listdir(output_dir) if
                   f.endswith(".xlsx") and "structured" not in f and current_year in f]

    if not excel_files:
        print("No Excel files for the current year found in the output directory.")
        return
    for file_name in excel_files:
        try:
            file_path = os.path.join(output_dir, file_name)
            df = pd.read_excel(file_path)
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
