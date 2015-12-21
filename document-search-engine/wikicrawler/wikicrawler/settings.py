# -*- coding: utf-8 -*-

# Scrapy settings for wikicrawler project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#

BOT_NAME = 'wikicrawler'

SPIDER_MODULES = ['wikicrawler.spiders']
NEWSPIDER_MODULE = 'wikicrawler.spiders'

EXTENSIONS = {
    'scrapy.contrib.closespider.CloseSpider': 0
}

CLOSESPIDER_PAGECOUNT = 10000
DEPTH_LIMIT = 4
CONCURRENT_REQUESTS_PER_IP = 1
DOWNLOAD_DELAY = 1.5

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'wikicrawler (+http://www.yourdomain.com)'
