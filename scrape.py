from firecrawl import FirecrawlApp
import tiktoken

urls_to_scrape = [
    "https://www.thebottomline.org.uk/summaries/icm/ttm/",
    "https://www.thebottomline.org.uk/summaries/hyperion/",
    "https://www.thebottomline.org.uk/summaries/icm/ttm2/",
    "https://www.thebottomline.org.uk/summaries/icm/evald/",
    "https://www.thebottomline.org.uk/summaries/icm/haca/",
]

steroid_urls = [
"https://www.thebottomline.org.uk/summaries/cape-cod/",
"https://www.thebottomline.org.uk/summaries/icm/meduri-1/",
"https://www.thebottomline.org.uk/summaries/icm/corticus/",
"https://www.thebottomline.org.uk/summaries/icm/annane/",
"https://www.thebottomline.org.uk/summaries/icm/meduri-2/",
"https://www.thebottomline.org.uk/summaries/icm/crash-1/",
"https://www.thebottomline.org.uk/summaries/icm/torres/",
"https://www.thebottomline.org.uk/summaries/icm/vanish/",
"https://www.thebottomline.org.uk/summaries/icm/hypress/",
"https://www.thebottomline.org.uk/summaries/icm/marik/",
"https://www.thebottomline.org.uk/summaries/icm/coiitss/",
"https://www.thebottomline.org.uk/blog/steroids-in-sepsis/",
"https://www.thebottomline.org.uk/summaries/icm/adrenal/",
"https://www.thebottomline.org.uk/summaries/icm/aprocchss/",
"https://www.thebottomline.org.uk/summaries/icm/recovery-covid-19-dexamethasone/",
]

dialysis_urls = [
    "https://www.thebottomline.org.uk/summaries/icm/starrt-aki/",
    "https://www.thebottomline.org.uk/summaries/icm/elain-follow-up/",
    "https://www.thebottomline.org.uk/summaries/icm/akiki/",
    "https://www.thebottomline.org.uk/summaries/icm/ideal-icu/",
    
]



def scrape(url):
    app = FirecrawlApp(api_key="your_api_key", api_url="http://raspberrypi.local:3002")
    scrape_url = url
    current_crawl = app.scrape_url(scrape_url)
    return current_crawl["markdown"]


docs = []

for url in dialysis_urls:
    print("Scraping:", url)
    scraped_markdown = scrape(url)
    # split everything before ### Leave a Reply
    scraped_markdown = scraped_markdown.split("### Leave a Reply")[0]
    docs.append(scraped_markdown)


# save the scraped markdown to a file
with open("data/dialysis_summaries.md", "w") as f:
    for doc in docs:
        f.write(doc)
        f.write("\n")
