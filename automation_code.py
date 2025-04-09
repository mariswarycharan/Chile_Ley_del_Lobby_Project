import schedule
import time
from data_scraping import scrape_data
from storygeneration import generate_narratives
from faiss_db import create_vector_store



# Function to execute the script
def run_script():
    print("Running scheduled script...")
    
    # Step 1 : Scrape data from the website
    print("Scraping data from the website...")
    scrape_data()
    print("Scraping completed.")
    
    # Step 2 : Generate narratives from the scraped data
    print("Generating narratives from the scraped data...")
    generate_narratives()
    print("Generating narratives completed.")
    
    # Step 3 : Create vector store
    print("Creating vector store...")
    create_vector_store()
    print("Vector store creation completed.")
    
    # Step 4 : push chnages to repository
    print("Pushing changes to repository...")
    
    
    print("Script execution completed.")

# Function to schedule the script
def schedule_script(day="sunday", exec_time="00:00"):
    # Clear any existing schedules
    schedule.clear()
    
    # Schedule the script
    getattr(schedule.every(), day).at(exec_time).do(run_script)
    print(f"Scheduled script to run every {day} at {exec_time}.")

# User input for day and time (Optional)
user_day = "Tuesday"
user_time = "00:00"

# Schedule the script based on user input
schedule_script(user_day, user_time)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute

