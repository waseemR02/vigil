import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import json
from urllib.robotparser import RobotFileParser

PARSERS = ["lxml", "html.parser", "html5lib"]
MAX_PAGES = 100  
KEYWORDS = ["cybersecurity", "security", "hacking", "infosec", "malware", "threat", "privacy"]  

def is_crawl_allowed(base_url):
    robots_url = urljoin(base_url, "robots.txt")
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch("*", base_url)
    except:
        return False  

def get_internal_links(url, base_url, visited):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        if "text/html" not in response.headers.get("Content-Type", ""):
            return set()
        
        soup = None
        for parser in PARSERS:
            try:
                soup = BeautifulSoup(response.text, parser)
                break
            except:
                continue
        if not soup:
            return set()
        
        links = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc == urlparse(base_url).netloc and full_url not in visited:
                if any(keyword in full_url.lower() for keyword in KEYWORDS):
                    links.add(full_url)
        return links
    except requests.RequestException:
        return set()

def scrape_content(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        if "text/html" not in response.headers.get("Content-Type", ""):
            return None
        
        response.encoding = response.apparent_encoding
        soup = None
        for parser in PARSERS:
            try:
                soup = BeautifulSoup(response.text, parser)
                break
            except:
                continue
        if not soup:
            return None
        
        title = soup.title.string.strip() if soup.title else "No Title"
        text = ' '.join(p.get_text(strip=True) for p in soup.find_all("p"))
        return {"url": url, "title": title, "content": text}
    except requests.RequestException:
        return None

def main():
    start_url = input("Enter the website URL to crawl and scrape: ")
    base_url = "{0.scheme}://{0.netloc}/".format(urlparse(start_url))
    if not is_crawl_allowed(base_url):
        print(f"Crawling not allowed by robots.txt on {base_url}")
        return
    
    domain_name = urlparse(start_url).netloc.replace("www.", "").replace('.', '_')
    output_json = f"scraped_data_{domain_name}.json"
    output_txt = f"links_{domain_name}.txt"
    
    visited = set()
    queue = [start_url]
    scraped_data = []
    collected_links = []
    
    while queue and len(scraped_data) < MAX_PAGES:
        current_url = queue.pop(0)
        if current_url in visited:
            continue
        
        print(f"Crawling: {current_url}")
        visited.add(current_url)
        collected_links.append(current_url)
        data = scrape_content(current_url)
        if data:
            scraped_data.append(data)
        new_links = get_internal_links(current_url, base_url, visited)
        queue.extend(new_links)
        time.sleep(1)  
    
    with open(output_txt, "w", encoding="utf-8") as file:
        for link in collected_links:
            file.write(link + "\n")
    
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(scraped_data, file, indent=4, ensure_ascii=False)
    
    print(f"\nScraping complete!")
    print(f"Links saved in: {output_txt}")
    print(f"Webpage data saved in: {output_json}")

if __name__ == "__main__":
    main()