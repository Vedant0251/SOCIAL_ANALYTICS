"""BrandPulse AI - Scrapy Cleaning and Export Pipelines."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import scrapy


class CleaningPipeline:
    """Cleans text fields, strips excess whitespace, and normalizes encodings."""

    def process_item(self, item: scrapy.Item, spider: scrapy.Spider) -> scrapy.Item:
        if "text" in item and item["text"]:
            text = str(item["text"]).strip()
            # Remove typographical curly quotes
            text = text.replace("“", '"').replace("”", '"')
            item["text"] = text

        if "author" in item and item["author"]:
            item["author"] = str(item["author"]).strip()

        if "tags" in item and isinstance(item["tags"], list):
            item["tags"] = [str(t).strip().lower() for t in item["tags"]]

        if "price" in item and item["price"]:
            price_str = str(item["price"]).replace("£", "").replace("$", "").strip()
            try:
                item["price"] = float(price_str)
            except ValueError:
                item["price"] = None

        item["scraped_at"] = datetime.now(timezone.utc).isoformat()
        return item


class CsvExportPipeline:
    """Exports scraped items directly to reports/tables/scraped_sandbox_quotes.csv."""

    def __init__(self) -> None:
        self.output_file: Path | None = None
        self.writer: Any = None
        self.file_handle: Any = None

    def open_spider(self, spider: scrapy.Spider) -> None:
        # Determine output location relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        out_dir = project_root / "reports" / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = out_dir / "scraped_sandbox_quotes.csv"

        self.file_handle = open(self.output_file, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file_handle)
        self.writer.writerow(["text", "author", "tags", "scraped_at"])

    def process_item(self, item: scrapy.Item, spider: scrapy.Spider) -> scrapy.Item:
        if self.writer:
            tags_str = ";".join(item.get("tags", [])) if isinstance(item.get("tags"), list) else ""
            self.writer.writerow([
                item.get("text", ""),
                item.get("author", ""),
                tags_str,
                item.get("scraped_at", ""),
            ])
            self.file_handle.flush()
        return item

    def close_spider(self, spider: scrapy.Spider) -> None:
        if self.file_handle:
            self.file_handle.close()
