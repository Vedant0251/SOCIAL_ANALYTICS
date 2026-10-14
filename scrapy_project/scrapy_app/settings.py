"""BrandPulse AI - Scrapy Settings Configuration.

Strictly follows ethical web crawling guidelines:
- Respects robots.txt (ROBOTSTXT_OBEY = True)
- Polite AutoThrottle with exponential delays and concurrency capping
- Identifies itself with a clear, polite User-Agent
"""

BOT_NAME = "scrapy_app"

SPIDER_MODULES = ["scrapy_app.spiders"]
NEWSPIDER_MODULE = "scrapy_app.spiders"

# Ethical & Polite Crawling Rules
ROBOTSTXT_OBEY = True
CONCURRENT_REQUESTS = 4
DOWNLOAD_DELAY = 1.0  # Polite 1-second delay between requests

USER_AGENT = "BrandPulseAI-ResearchBot/1.0 (+https://github.com/Vedant0251/SOCIAL_ANALYTICS; academic/research polite crawler)"

# AutoThrottle settings
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 10.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
AUTOTHROTTLE_DEBUG = False

# Pipelines
ITEM_PIPELINES = {
    "scrapy_app.pipelines.CleaningPipeline": 300,
    "scrapy_app.pipelines.CsvExportPipeline": 800,
}

# Disable cookies to maintain privacy and stateless requests
COOKIES_ENABLED = False

# Log level
LOG_LEVEL = "INFO"
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"
