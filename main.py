import pickle
import os
from kmeans import *
from preprocess import *


N = 100
webfilename = "./data/webpages.pickle"

urls = ["https://en.wikipedia.org/wiki/Geography_of_the_United_States",
        "https://en.wikipedia.org/wiki/Geography_of_Germany",
        "https://en.wikipedia.org/wiki/Geography_of_Japan",
        "https://www.cnbc.com/2017/12/06/facebook-will-lead-a-surge-in-tech-stocks-next-year-with-a-30-percent-gain-evercore-isi-says.html",
        "https://www.cnbc.com/2017/12/06/stocks-making-the-biggest-moves-premarket-googl-fb-baba-avav-unh-hd-more.html",
        "https://www.cnbc.com/2017/12/05/snap-shares-jump-after-barclays-says-turning-point-coming-in-2018.html",
        "https://www.cnbc.com/2017/12/06/aerovironment-shares-jump-more-than-20-percent-after-announcing-booming-sales.html",
        "https://www.rogerebert.com/reviews/the-dancer-2017",
        "https://www.rogerebert.com/reviews/great-movie-the-ballad-of-narayama-1958",
        "https://www.rogerebert.com/reviews/the-breadwinner-2017"]
reload = True

def main():
    # load url content if already downloaded,
    # otherwise use scraper to scrape contents on the fly.
    if os.path.isfile(webfilename) and reload:
        with open(webfilename, 'rb') as file:
            rawtexts = pickle.load(file)
    else:
        rawtexts = []
        for url in urls:
            rawtexts += [scrape_website(url)]

        with open(webfilename, "w") as file:
            pickle.dump(rawtexts, file)

    # convert raw text to vectored. each feature is hashed.
    text_vectors = []
    for text in rawtexts:
        features = text_to_words(text)
        text_vectors += [ words_to_vector(features, ndim = N) ]

    #print(text_vectors)

    # apply k means algorithm.
    clusters, labels = kmeans(text_vectors, 3)

    # print(labels)
    # Now show content from which url belongs to which cluster
    for clusterindex in labels:
        print("cluster:" + str(clusterindex) + "\n")
        for urlindex in labels[clusterindex]:
            print("\t" + urls[urlindex])

# Now call main
main()









