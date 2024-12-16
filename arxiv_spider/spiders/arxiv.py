import scrapy
from datetime import datetime, timedelta
import re
import time
from scrapy.loader import ItemLoader
from scrapy.http import FormRequest
from arxiv_spider.items import ArxivItem
import logging


logging.basicConfig(
    filename='spider.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class ArxivSpider(scrapy.Spider):
    name = "arxiv"
    
    def __init__(self, search_term='', from_date=None, to_date=None, **kwargs):
        super().__init__(**kwargs)
        
        self.search_term = search_term.replace(' ', '+')

        if from_date is None:
            self.from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        else:
            self.from_date = from_date
            
        if to_date is None:
            self.to_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.to_date = to_date
            
        self.start = 0
        self.url = f"https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term={self.search_term}&terms-0-field=title&classification-computer_science=y&classification-physics_archives=all&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date={self.from_date}&date-to_date={self.to_date}&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first"
        
        self.start_urls = [self.url]
        self.total_papers = 0
        
    def parse(self, response):
        logging.info(f"response: {response}")
        for paper in response.css("li.arxiv-result"):
            self.total_papers +=1
            
            new = ItemLoader(item=ArxivItem(), selector=paper) 

            ID = paper.css("p.list-title a::text").extract_first().strip('arXiv:')
            title = ''.join(paper.css("p.title").xpath("text() | span[@class='search-hit mathjax']/text()").extract()).strip()
            authors = paper.css("p.authors").css("a::text").extract()
            primary_cat =  paper.css("span.tag::text").extract_first()
            abstract = ' '.join([sent if sent!=' ' else self.\
                                            search_query_or for sent in  paper.css("span.abstract-full::text").\
                                            extract()]).strip('\n ')
            
            # comments = paper.css('p.comments span::text').extract()
            # if len(comments)>1:
            #     comments = comments[1]
            # else:
            #     comments = ''
                
            # journal = paper.css("p.comments::text").extract()
            # if len(journal)>0:
            #     journal = journal[-1].strip('\n')
            # else:
            #     journal = ''
            abs_page = paper.css("p.list-title a::attr(href)").extract_first()
            pdf_page = paper.css("p.list-title a::attr(href)").extract()[1]
            
            new.add_value('ID', ID)
            new.add_value('title', title)
            new.add_value('author', authors)
            new.add_value('primary_cat', primary_cat)
            new.add_value('abstract', abstract)
            new.add_value('link', abs_page)
            new.add_value('pdf', pdf_page)
            logging.info(f"paper: {new.load_item()}")
            
            yield scrapy.Request(abs_page, callback=self.parse_abs_page, dont_filter = True, meta={'item':new})            
        
        # scrape next page until one exist
        next_page = response.xpath("//nav//a[contains(@class, 'pagination-next')]")
        if next_page:
            next_page_url = next_page.xpath("@href").extract_first()
            if next_page_url:
                yield response.follow(next_page_url, callback=self.parse)
            
        print(f"Total papers scraped: {self.total_papers}")

    
    def parse_abs_page(self, response):        
        new = ItemLoader(item=ArxivItem(), response=response, parent=response.meta['item']) 
        # all arXiv categories
        other_cat_full_cont = response.css('td[class*=subjects]').extract()[0].split('</span>;')
        if len(other_cat_full_cont)>1:
            other_cats = other_cat_full_cont[1]
            other_cats_list = [x.strip('\(').strip('\)') for x in re.findall('\(.*?\)', other_cats)]
        else: other_cats_list = []
            
        main_cat = re.findall('\(.*?\)', response.css('div.metatable span::text').extract()[0])[0].strip('\(').strip('\)')
        all_cats =[main_cat]+other_cats_list
        new.add_value('all_cat', all_cats)
        
        # submission date
        new.add_value('date', response.css('div.submission-history::text').extract()[-2])
        
        yield new.load_item()
        
