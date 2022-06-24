from review_classifier_service import ReviewClassifierService

# Initialise call to service 
service = ReviewClassifierService()


positive_sample = "It was great"
negative_sample = "It was amazing"      # try with exclamation marks, funny results 
print(service.classify(positive_sample))
print(service.classify(negative_sample))

