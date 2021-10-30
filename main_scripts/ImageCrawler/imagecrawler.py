from DataOps import selenium_driver

search_term = 'over ripe banana'

image_crawler = selenium_driver()
image_crawler.search_and_download(search_term)

# TODO: create a parser json file and extract phrases here
#   create a list of phrases to be extracted and set up
#   manually clean up images which are ambiguous
