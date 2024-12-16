import scrapy
from scrapy.crawler import CrawlerProcess
from arxiv_spider.spiders.arxiv import ArxivSpider
import argparse
import os
from datetime import datetime, timedelta

def main(search_term, from_date, to_date, output_dir):
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'FEEDS': {
            os.path.join(output_dir, f'[{search_term}]_[{from_date}]_[{to_date}].json'): {
                'format': 'json',
                'encoding': 'utf8',
                'store_empty': False,
                'indent': 4,
            },
        }
    })

    process.crawl(ArxivSpider, search_term=search_term, from_date=from_date, to_date=to_date)
    process.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Arxiv Spider with parameters.')
    parser.add_argument('search_term', type=str, nargs='?', default='', help='Search term for arXiv')
    parser.add_argument('from_date', type=str, nargs='?', default=(datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'), help='Start date for search (YYYY-MM-DD)')
    parser.add_argument('to_date', type=str, nargs='?', default=datetime.now().strftime('%Y-%m-%d'), help='End date for search (YYYY-MM-DD)')
    parser.add_argument('output_dir', type=str, nargs='?', default='./spider_outputs', help='Directory to save the output file')
    
    args = parser.parse_args()
    main(args.search_term, args.from_date, args.to_date, args.output_dir)