import requests
import json
from bs4 import BeautifulSoup
import time
import ollama
from functions import clean_html

DATA_DIR = "data_by_file_md"

# Function to get the HTML content of a webpage
def get_page_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve page: {url}")
        return None

# Function to scrape the summaries from a page
def scrape_summaries_from_page(page_content):
    soup = BeautifulSoup(page_content, 'html.parser')

    # Find all article blocks (Adjust the class name based on actual site structure)
    for article in soup.find_all('article'):
        title = article.find('h2').get_text(strip=True)  # Extract the title
        link = article.find('a')['href']  # Extract the link (Continue reading link)
        summary = article.find('p').get_text(strip=True) if article.find('p') else "No summary available"  # Extract the summary

        # Scrape full article content from the 'Continue reading' link
        full_article_content = scrape_full_article(link)

        content ={
            'title': title,
            'link': link,
            'summary': summary,
            'full_content': full_article_content
        }
        with open(f'{DATA_DIR}/{title.replace(" ", "_").strip()}.json', 'w') as json_file:
          json.dump(content, json_file, indent=4)

# Function to scrape the full article from the "Continue reading" page
def scrape_full_article(article_url):
    article_page = get_page_content(article_url)
    if not article_page:
        return "Unable to retrieve full article"

    soup = BeautifulSoup(article_page, 'html.parser')

    # Adjust this based on the structure of the full article page
    article_content = soup.find('div', class_='entry clearfix')  # Assuming 'entry-content' holds the article text
    if article_content:
        return article_content.get_text()
    else:
        return "Full content not available"

# Function to find the next page link
def get_next_page_url(soup):
    next_button = soup.find('a', class_='next')  # Adjust class based on site's pagination structure
    if next_button:
        return next_button['href']
    return None

# Main scraping function
def scrape_all_pages(base_url, scraper=None):
    next_page_url = base_url

    while next_page_url:
        print(f"Scraping page: {next_page_url}")
        page_content = get_page_content(next_page_url)
        if page_content:
            soup = BeautifulSoup(page_content, 'html.parser')
            scrape_summaries_from_page(page_content)

            # Get next page URL
            next_page_url = get_next_page_url(soup)

            # Delay to avoid overwhelming the server
            time.sleep(2)
        else:
            break

# URL of the ICM summaries page
base_url = "https://www.thebottomline.org.uk/category/summaries/icm/"

# Start scraping
scrape_all_pages(base_url)
