from DataOps import selenium_driver

search_term = ['over riped bananas','unriped bananas']

image_crawler = selenium_driver()
for items in search_term:
    image_crawler.search_and_download(items)

# TODO: create a parser json file and extract phrases here
#   create a list of phrases to be extracted and set up
#   manually clean up images which are ambiguous
