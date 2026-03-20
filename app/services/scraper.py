from bs4 import BeautifulSoup
import requests


def extract_article(url: str) -> dict[str, str]:
    response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    paragraphs = [paragraph.get_text(" ", strip=True) for paragraph in soup.find_all("p")]
    article_text = " ".join(fragment for fragment in paragraphs if fragment)
    if len(article_text) < 120:
        raise ValueError("Unable to extract enough article text from the provided URL.")

    return {"title": title, "text": article_text, "url": url}
