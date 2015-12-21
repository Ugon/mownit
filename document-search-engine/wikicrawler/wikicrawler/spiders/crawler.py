from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors import LinkExtractor
import scrapy.link
import os
import re


class SeeAlsoExtractor(LinkExtractor):
    def extract_links(self, response):
        try:
            body = response.xpath("//div[@id='mw-content-text']").extract()[0].encode('utf-8')
            see_also = body.split('id="See_also"')[1].split('id="References"')[0].split('id="Notes"')[0]
            links = [link for link in re.findall('<a.*?href="(.*?)".*?>', see_also) if link.startswith('/wiki/')]
            links = map(lambda x: x.replace('/', '-').replace('-wiki-', '/wiki/'), links)
            links = filter(lambda x: not x.startswith('/wiki/Template'), links)
            links = filter(lambda x: not x.startswith('/wiki/File'), links)
            links = filter(lambda x: not x.startswith('/wiki/Portal'), links)
            links = filter(lambda x: not x.startswith('/wiki/Wikipedia'), links)
            links = filter(lambda x: not x.startswith('/wiki/Special'), links)
            links = filter(lambda x: not x.startswith('/wiki/Category'), links)
            links = filter(lambda x: not x.startswith('/wiki/User'), links)
            links = filter(lambda x: not x.startswith('/wiki/MediaWiki'), links)
            links = filter(lambda x: not x.startswith('/wiki/Help'), links)
            links = filter(lambda x: not x.startswith('/wiki/Book'), links)
            links = filter(lambda x: not x.startswith('/wiki/Draft'), links)
            links = filter(lambda x: not x.startswith('/wiki/TimedText'), links)
            links = filter(lambda x: not x.startswith('/wiki/Module'), links)
            links = filter(lambda x: not x.startswith('/wiki/Topic'), links)
            return [scrapy.link.Link('http://en.wikipedia.org' + link) for link in links]
        except Exception:
            return []


class WikipediaSpider(CrawlSpider):
    name = 'wikicrawler'
    start_urls = ['http://en.wikipedia.org/wiki/Central_processing_unit',
                  'http://en.wikipedia.org/wiki/Graphics_processing_unit',
                  'http://en.wikipedia.org/wiki/Random-access_memory',
                  'http://en.wikipedia.org/wiki/Cache_(computing)',
                  'http://en.wikipedia.org/wiki/Serial_ATA',
                  'http://en.wikipedia.org/wiki/Computer',
                  'http://en.wikipedia.org/wiki/Transistor',
                  'http://en.wikipedia.org/wiki/Capacitor',
                  'http://en.wikipedia.org/wiki/Resistor',
                  'http://en.wikipedia.org/wiki/Blu-ray_Disc',
                  'http://en.wikipedia.org/wiki/Solid-state_drive',
                  'http://en.wikipedia.org/wiki/Hard_disk_drive',
                  'http://en.wikipedia.org/wiki/Haskell_(programming_language)',
                  'http://en.wikipedia.org/wiki/Python_(programming_language)',
                  'http://en.wikipedia.org/wiki/Scala_(programming_language)',
                  'http://en.wikipedia.org/wiki/Java_(programming_language)',
                  'http://en.wikipedia.org/wiki/Java_(software_platform)',
                  'http://en.wikipedia.org/wiki/Java_virtual_machine',
                  'http://en.wikipedia.org/wiki/GNU_Octave',
                  'http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms',
                  'http://en.wikipedia.org/wiki/Algorithm',
                  'http://en.wikipedia.org/wiki/Parallel_computing',
                  'http://en.wikipedia.org/wiki/Information_technology',
                  'http://en.wikipedia.org/wiki/Graph_(mathematics)',
                  'http://en.wikipedia.org/wiki/Breadth-first_search',
                  'http://en.wikipedia.org/wiki/Bitonic_sorter',
                  'http://en.wikipedia.org/wiki/Facebook',
                  'http://en.wikipedia.org/wiki/League_of_Legends',
                  'http://en.wikipedia.org/wiki/Video_game',
                  'http://en.wikipedia.org/wiki/Peer-to-peer_file_sharing',
                  'http://en.wikipedia.org/wiki/BitTorrent',
                  'http://en.wikipedia.org/wiki/Computer_file',
                  'http://en.wikipedia.org/wiki/Heap_(data_structure)',
                  'http://en.wikipedia.org/wiki/Memory_management',
                  'http://en.wikipedia.org/wiki/Disk_storage',
                  'http://en.wikipedia.org/wiki/Data',
                  'http://en.wikipedia.org/wiki/Matrix_(mathematics)',
                  'http://en.wikipedia.org/wiki/Vector_(mathematics_and_physics)',
                  'http://en.wikipedia.org/wiki/Wi-Fi',
                  'http://en.wikipedia.org/wiki/Computer_network',
                  'http://en.wikipedia.org/wiki/Internet']
    allowed_domains = ['en.wikipedia.org']
    rules = [Rule(SeeAlsoExtractor(), callback='parse_article', follow=True)]

    def __init__(self, *args, **kwargs):
        super(WikipediaSpider, self).__init__(*args, **kwargs)
        if not os.path.exists('../various'):
            os.makedirs('../various')
        if not os.path.exists('../html-articles'):
            os.makedirs('../html-articles')
        if not os.path.exists('../various'):
            os.makedirs('../various')
        self.url_log = open('../various/urls.csv', 'w+')

    def parse_start_url(self, response):
        self.parse_article(response)

    def parse_article(self, response):
        try:
            url = response.url
            name = response.xpath("//h1[@id='firstHeading']/text()").extract()[0].encode('utf-8')
            content = response.xpath("//div[@id='mw-content-text']").extract()[0].encode('utf-8')
        except Exception:
            return

        try:
            f = open('../html-articles/' + name + '.html', 'w')
            f.write(content)
            f.close()
        except Exception:
            return

        try:
            self.url_log.write(name + "; " + url + '\n')
        except Exception:
            try:
                os.remove('../html-articles/' + name + '.html')
            finally:
                return

    def closed(self, reason):
        self.url_log.close()

