import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

def fetch_today_news(ticker: str, count: int = 10) -> str:
    """
    Lekéri a legfrissebb híreket a Google News RSS Feed-ből az adott részvényhez.
    Külső csomagot nem igényel, beépített Python modulokat használ.
    """
    query = urllib.parse.quote(f"{ticker} stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        response = urllib.request.urlopen(req)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        news_list = []
        for item in root.findall('.//item')[:count]:
            title = item.find('title').text
            pubDate = item.find('pubDate').text
            news_list.append(f"- {title} (Közzétéve: {pubDate})")
            
        if not news_list:
            return "Nincsenek friss hírek a megadott tickerhez."
            
        return "\n".join(news_list)
        
    except Exception as e:
        return f"Hiba a hírek lekérésekor: {e}"

if __name__ == "__main__":
    print(f"Teszt hívás (NVDA):")
    print(fetch_today_news("NVDA", count=3))
