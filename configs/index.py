from team17.data.indexing import IndexConfig, index_youtube_urls

config = IndexConfig(min_duration=10 * 60)
index_youtube_urls(config)
