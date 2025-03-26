import feedparser

def get_google_news():
    """Fetches latest news from Google RSS Feed."""
    url = "https://news.google.com/rss"
    feed = feedparser.parse(url)

    news_list = []
    for entry in feed.entries[:10]:  # Top 10 news
        news_list.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
            "source": entry.source.title if "source" in entry else "Unknown"
        })
    
    return news_list

if __name__ == "__main__":
    news = get_google_news()
    print("\n📌 Latest Google News:")
    for n in news:
        print(f"- {n['title']} ({n['source']})")