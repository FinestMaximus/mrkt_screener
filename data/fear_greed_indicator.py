import requests
from bs4 import BeautifulSoup
import re
import random
from utils.logger import info, debug, warning, error


class FearGreedIndicator:
    """Class responsible for analyzing market sentiment through fear and greed indicators"""

    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
            # Additional user agents can be included as needed
        ]

    def get_random_user_agent(self):
        """Return a random user agent from the list"""
        return random.choice(self.user_agents)

    def fetch_market_sentiment(self):
        """
        Fetches market sentiment data from a given URL.

        Args:
            url (str): URL to fetch market sentiment data from

        Returns:
            tuple: (percentage, sentiment, color_code) or (None, None, None) if failed
        """
        try:
            url = "https://pyinvesting.com/fear-and-greed/"
            debug(
                "Initiating request to fetch market sentiment from URL: %s",
                url,
            )

            # Use a random user agent to reduce chance of being blocked
            headers = {"User-Agent": self.get_random_user_agent()}
            response = requests.get(url, headers=headers)

            debug("Received response with status code: %s", response.status_code)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            debug("Parsed HTML content successfully.")
            selector = (
                "div > div > div:nth-of-type(2) > div:nth-of-type(1) > p:nth-of-type(2)"
            )
            extracted_text = soup.select_one(selector).text
            debug("Extracted text: %s", extracted_text)
            match = re.search(r"\d+%", extracted_text)
            if match:
                percentage_value = float(match.group().strip("%"))
                debug("Extracted percentage value: %s", percentage_value)

                if percentage_value >= 75:
                    sentiment = "Extreme Greed"
                    color_code = "red"
                elif 60 <= percentage_value < 75:
                    sentiment = "Greed 😨"
                    color_code = "orange"
                elif 40 <= percentage_value < 60:
                    sentiment = "Neutral 😐"
                    color_code = "white"
                elif 25 <= percentage_value < 40:
                    sentiment = "Fear 😏"
                    color_code = "yellow"
                else:
                    sentiment = "Extreme Fear"
                    color_code = "green"

                debug(
                    "Determined sentiment: %s, with color code: %s",
                    sentiment,
                    color_code,
                )
                return match.group(), sentiment, color_code
            else:
                debug("No percentage found in the extracted text.")
                return None, None, None
        except requests.RequestException as e:
            error(
                "RequestException occurred - Error: %s",
                e,
            )
            return None, None, None
