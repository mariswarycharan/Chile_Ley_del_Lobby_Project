import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
import random

# Base URL
base_url = "https://www.news-medical.net/condition/Multiple-Sclerosis"

# User-Agent Header to Mimic Browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}

# List to Store Articles Data
articles_data = []

# Fetch the Main Page
response = requests.get(base_url, headers=headers)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")

    # Locate all "Read More" links (Overview, Featured Articles, Latest News)
    sections = soup.find_all("div", class_="hub-page-content")
    article_links = []

    # Collect all article links
    for section in sections:
        for article_link in section.find_all("a", href=True, string="Read More"):
            article_url = f"https://www.news-medical.net{article_link['href']}"
            article_links.append(article_url)

    # Use tqdm for progress tracking while scraping each article
    for article_url in tqdm(article_links, desc="Scraping Articles", unit="article"):
        # Fetch the article page
        article_response = requests.get(article_url, headers=headers)
        if article_response.status_code == 200:
            article_soup = BeautifulSoup(article_response.text, "html.parser")

            # Extract Details
            title = article_soup.find("h1").get_text(strip=True) if article_soup.find("h1") else "No Title"

            # Extract Author Name and Description
            author_section = article_soup.find("div", class_="author")
            if author_section:
                # Extract Author Name
                author = author_section.find("a").get_text(strip=True) if author_section.find("a") else "No Author"

                # Extract Author Description
                author_description = author_section.get_text(strip=True).replace(f"Written by {author}", "").strip()
                author_description = author_description.split("Reviewed by")[
                    0].strip()  # Remove "Reviewed by" if present
            else:
                author = "No Author"
                author_description = "No Author Description"

            # Extract Reviewer
            reviewer_section = article_soup.find("span", string="Reviewed by")
            reviewer = (
                reviewer_section.parent.get_text(strip=True)
                if reviewer_section
                else "No Reviewer"
            )

            # Extract Description
            description = article_soup.find("meta", attrs={"name": "description"})["content"] if article_soup.find(
                "meta", attrs={"name": "description"}) else "No Description"

            # Extract Full Content
            content = " ".join([p.get_text(strip=True) for p in article_soup.find_all("p")])

            # Extract Sources
            sources = [source.get("href") for source in article_soup.find_all("a", href=True) if
                       "http" in source.get("href")]

            # Extract Further Reading
            further_reading = [reading.get_text(strip=True) for reading in
                               article_soup.find_all("a", class_="further-reading-link")]

            # Citation (Optional: Expand dropdown via additional scraping logic if dynamic)
            citation = "No Citation Available"
            citation_section = article_soup.find("div", class_="citation")
            if citation_section:
                citation = citation_section.get_text(strip=True)

            # Store the Article Data
            articles_data.append({
                "Title": title,
                "Author": author,
                "Author Description": author_description,
                "Reviewed By": reviewer,
                "Description": description,
                "Content": content,
                "Sources": sources,
                "Further Reading": further_reading,
                "Citation": citation
            })

            # Add a Random Delay to Avoid Getting Blocked
            time.sleep(random.uniform(1, 3))

else:
    print(f"Failed to fetch the main page. Status code: {response.status_code}")

# Save the Data to Excel
df = pd.DataFrame(articles_data)
df.to_excel("multiple_sclerosis_articles.xlsx", index=False)

print("Scraping complete. Data saved to 'multiple_sclerosis_articles.xlsx'.")
