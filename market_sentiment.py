import re
import logging
import requests
from bs4 import BeautifulSoup


def fetch_market_sentiment(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        selector = (
            "div > div > div:nth-of-type(2) > div:nth-of-type(1) > p:nth-of-type(2)"
        )
        extracted_text = soup.select_one(selector).text
        match = re.search(r"\d+%", extracted_text)
        if match:
            percentage_value = float(match.group().strip("%"))

            if percentage_value >= 75:
                sentiment = "Extreme Greed"
                color_code = "red"
            elif 50 <= percentage_value < 75:
                sentiment = "Greed 😨"
                color_code = "orange"
            elif 25 <= percentage_value < 50:
                sentiment = "Fear 😏"
                color_code = "yellow"
            else:
                sentiment = "Extreme Fear"
                color_code = "green"

            return match.group(), sentiment, color_code
        else:
            logging.info("Failed to find the percentage in the extractor text.")
            return None, None, None
    else:
        logging.error(
            "Failed to retrieve the webpage - Status code: %s", response.status_code
        )
        return None, None, None
