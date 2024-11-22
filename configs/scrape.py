from team17.data.scraping import ScrapingConfig, scrape_youtube_urls

config = ScrapingConfig(max_workers=2, chunk_duration=40)
scrape_youtube_urls(config)
