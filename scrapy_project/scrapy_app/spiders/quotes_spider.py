"""BrandPulse AI - Quotes Sandbox Spider (Ethical Scraping Demonstration).

Crawls the dedicated public sandbox https://quotes.toscrape.com to demonstrate
spider lifecycle, CSS/XPath extraction, item pipelines, and pagination handling.
"""

from __future__ import annotations

from typing import Any, Generator
import scrapy
from scrapy_app.items import QuoteItem


class QuotesSpider(scrapy.Spider):
    name = "quotes"
    allowed_domains = ["quotes.toscrape.com"]
    start_urls = ["https://quotes.toscrape.com/"]

    custom_settings = {
        "CLOSESPIDER_ITEMCOUNT": 30,  # Polite sample cap for pipeline demonstration
    }

    def parse(self, response: Any) -> Generator[QuoteItem | scrapy.Request, None, None]:
        for quote_div in response.css("div.quote"):
            item = QuoteItem()
            item["text"] = quote_div.css("span.text::text").get()
            item["author"] = quote_div.css("small.author::text").get()
            item["tags"] = quote_div.css("div.tags a.tag::text").getall()
            yield item

        next_page = response.css("li.next a::attr(href)").get()
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)
