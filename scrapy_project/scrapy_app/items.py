"""BrandPulse AI - Scrapy Items & Data Definitions."""

from __future__ import annotations

import scrapy


class QuoteItem(scrapy.Item):
    """Container for quote items scraped from sandbox target."""

    text = scrapy.Field()
    author = scrapy.Field()
    tags = scrapy.Field()
    scraped_at = scrapy.Field()


class BookItem(scrapy.Item):
    """Container for book reviews and prices from sandbox target."""

    title = scrapy.Field()
    price = scrapy.Field()
    rating = scrapy.Field()
    availability = scrapy.Field()
    scraped_at = scrapy.Field()
