from DataOps import selenium_driver

searchterm='over ripe banana'

image_crawler=selenium_driver()
DRIVER_PATH='C:/Users/saksh/PycharmProjects/FreshPrice/FreshPrice/Resources/Selenium/chromedriver.exe'
image_crawler.search_and_download(searchterm,DRIVER_PATH)

#TODO: create a parser json file and extract phrases here
#   create a list of phrases to be extracted and set up
#   manually clean up images which are ambiguous
