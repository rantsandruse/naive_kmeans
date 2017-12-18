This project is structured as follows:
1. data/webpages.pickle: downloaded web contents.
2. kmeans.py: kmeans algorithm implementation and associated functions
3. preprocess.py: scrape web pages and convert them into vectors
4. main.py: main function to be executed

The webpages are scraped from three disparate sources:
 1. wikipedia country geography
 2. CNBC financial news
 3. Roger Ebert movie review

Text cleaning includes the following steps:
 1. Parse through beautifulsoup
 2. tokenize all words using NLTK word tokenizer
 3. include alphanumeric words only


The output of main.py is given below. It shows that we are able to separate the webpages into its sources using kmeans.
The clusters(0,1,2) below represent CNBC financial news, movie reviews and wikipedia geography pages respectively.
 cluster:0

	https://www.cnbc.com/2017/12/06/facebook-will-lead-a-surge-in-tech-stocks-next-year-with-a-30-percent-gain-evercore-isi-says.html
	https://www.cnbc.com/2017/12/06/stocks-making-the-biggest-moves-premarket-googl-fb-baba-avav-unh-hd-more.html
	https://www.cnbc.com/2017/12/05/snap-shares-jump-after-barclays-says-turning-point-coming-in-2018.html
	https://www.cnbc.com/2017/12/06/aerovironment-shares-jump-more-than-20-percent-after-announcing-booming-sales.html
cluster:1

	https://www.rogerebert.com/reviews/the-dancer-2017
	https://www.rogerebert.com/reviews/great-movie-the-ballad-of-narayama-1958
	https://www.rogerebert.com/reviews/the-breadwinner-2017
cluster:2

	https://en.wikipedia.org/wiki/Geography_of_the_United_States
	https://en.wikipedia.org/wiki/Geography_of_Germany
	https://en.wikipedia.org/wiki/Geography_of_Japan




