build:
	cd wikicrawler ; scrapy crawl wikicrawler ; cd ..
	python ./preprocessor.py
	python ./calculator.py

clean:
	rm -fr bags-of-words
	rm -fr html-articles
	rm -fr matrices
	rm -fr various

# run with ./server with options of your choosing