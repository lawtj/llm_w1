from firecrawl import FirecrawlApp
import tiktoken

urls_to_scrape = ['https://www.thebottomline.org.uk/summaries/icm/ttm/',
                  'https://www.thebottomline.org.uk/summaries/hyperion/',
                  'https://www.thebottomline.org.uk/summaries/icm/ttm2/',
                  'https://www.thebottomline.org.uk/summaries/icm/evald/',
                  'https://www.thebottomline.org.uk/summaries/icm/haca/']

def scrape(url):
    app = FirecrawlApp(api_key='your_api_key', api_url='http://localhost:3002')
    scrape_url = url
    current_crawl = app.scrape_url(scrape_url)
    return current_crawl['markdown']

docs = []

for url in urls_to_scrape:
    print('Scraping:', url)
    scraped_markdown = scrape(url)
    # split everything before ### Leave a Reply
    scraped_markdown = scraped_markdown.split('### Leave a Reply')[0]
    docs.append(scraped_markdown)


#save the scraped markdown to a file
with open('scraped_markdown.md', 'w') as f:
    for doc in docs:
        f.write(doc)
        f.write('\n')

