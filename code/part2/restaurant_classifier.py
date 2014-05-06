from __future__ import division
import sys
import csv
import argparse
import json
from collections import defaultdict
from operator import itemgetter

from pprint import pprint

import util

import pdb

import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import cross_validation



from tokenizer import Tokenizer

from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

'''
JSON Structure

Businesses:
    {
        'type': 'business',
        'business_id': (encrypted business id),
        'name': (business name),
        'neighborhoods': [(hood names)],
        'full_address': (localized address),
        'city': (city),
        'state': (state),
        'latitude': latitude,
        'longitude': longitude,
        'stars': (star rating, rounded to half-stars),
        'review_count': review count,
        'categories': [(localized category names)]
        'open': True / False (corresponds to closed, not business hours),
        'hours': {
            (day_of_week): {
                'open': (HH:MM),
                'close': (HH:MM)
            },
            ...
        },
        'attributes': {
            (attribute_name): (attribute_value),
            ...
        },
    }

    Important fields:
        business['categories'] should contain 'Restaurants'
        business['attributes']['Price Range']

Reviews:
    {
        'type': 'review',
        'business_id': (encrypted business id),
        'user_id': (encrypted user id),
        'stars': (star rating, rounded to half-stars),
        'text': (review text),
        'date': (date, formatted like '2012-03-14'),
        'votes': {(vote type): (count)},
    }

    Important fields:
        review['business_id']
        review['text']
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reviews', required=True, help='Path to review data')
    parser.add_argument('-b', '--businesses', required=True, help='Path to business data')
    parser.add_argument('-cats', '--categories', required=True, help='File of possible retaurant categories')
    parser.add_argument('-stop', required=True, help='Stopwords file')
    parser.add_argument('-top', type=int, help='Number of top features to show')
    parser.add_argument('-first', type=int, help='Number of reviews to use')
    parser.add_argument('-c', '--classifier', help='Classifier to use. Options are: RF, NB, LR')
    opts = parser.parse_args()
    ##

    ## Building training set
    # Initialize CountVectorizer
    tokenizer = Tokenizer(opts.stop)
    vectorizer = CountVectorizer(binary=True, lowercase=True, decode_error='replace', tokenizer=tokenizer)


    # load business and categories
    all_categories = set(line.strip() for line in open(opts.categories))

    bids_to_categories = {}
    businesses_file = open(opts.businesses)
    for line in businesses_file:
        business = json.loads(line)
        categories = business['categories']
        if 'Restaurants' in categories:
            categories = filter(lambda x: x in all_categories, categories)
            if len(categories) != 0:
                bids_to_categories[business['business_id']] = categories

    #pprint(bids_to_categories)

    # Load training review text
    print "-- Extracting Features --"
    reviews = []
    labels = []
    review_file = open(opts.reviews)
    i = 0
    for line in review_file:
        if opts.first != None and i > opts.first: break
        review = json.loads(line)
        # check if this is a review for one of the restuarants with labeled categories
        if review['business_id'] in bids_to_categories:
            reviews.append(review['text'])
            categories = bids_to_categories[review['business_id']]
            labels.append(categories)

    # shrink dataset
    # if opts.first is not None:
    #     reviews = reviews[:opts.first]
    #     labels = labels[:opts.first]

    # Get training features using vectorizer
    train_features = vectorizer.fit_transform(reviews)

    # Transform training labels to numpy array (numpy.array)
    train_labels = numpy.array(labels)
    ############################################################


    ##### TRAIN THE MODEL ######################################
    print "-- Training Classifier --"
    if opts.classifier == 'RF':
        classifier = RandomForestClassifier(n_jobs = -1, verbose = 1)
        train_features = train_features.toarray()
    elif opts.classifier == 'NB':
        classifier = OneVsRestClassifier(BernoulliNB())
    elif opts.classifier == 'SVC':
        classifier = OneVsRestClassifier(LinearSVC())
    else:
        print "Invalid classifier " + str(opts.classifier)
        return
    classifier.fit(train_features, train_labels)
    ############################################################


    ###### VALIDATE THE MODEL ##################################
    # Print training mean accuracy using 'score'
    print "-- Validation --"
    # print "Mean accuracy on training data:", classifier.score(train_features, train_labels)

    predicted_labels = classifier.predict(train_features)
    print classification_report(train_labels, predicted_labels)

    # Perform 10 fold cross validation (cross_validation.cross_val_score) with scoring='accuracy'
    # and print the mean score and std deviation
    scores = cross_validation.cross_val_score(classifier, train_features, train_labels,
        scoring='accuracy', cv=10, n_jobs=-1) # passing integer for cv uses StratifiedKFold where k = integer

    # print "Cross validation mean score:", numpy.mean(scores)
    # print "Cross validation standard deviation:", numpy.std(scores)
    ############################################################


    ##### EXAMINE THE MODEL ####################################
    if opts.top is not None:
        # print top n most informative features for positive and negative classes
        print "Top", opts.top, "most informative features:"
        print_top(opts.top, vectorizer, classifier)
    ############################################################


    ##### TEST THE MODEL #######################################
    # print "-- Testing --"
    # if opts.test is None:
    #     # Test the classifier on one sample test tweet
    #     # Tim Kraska 10:43 AM - 5 Feb 13
    #     test_tweet = 'Water dripping from 3rd to 1st floor while the firealarm makes it hard to hear anything. BTW this is the 2nd leakage.  Love our new house'

    #     sample_test_features = vectorizer.transform([test_tweet]) # extract features

    #     # Print the predicted label of the test tweet
    #     print "Test tweet predicted label:", classifier.predict(sample_test_features)[0]

    #     # Print the predicted probability of each label.
    #     if opts.classifier != 'svm':
    #         class_probs = classifier.predict_proba(sample_test_features)[0]
    #         print "Probability of label 0:", class_probs[0]
    #         print "Probability of label 1:", class_probs[1]
    #     else:
    #         print "Confidence score for test tweet:", classifier.decision_function(sample_test_features)[0]

    # else:
    #     # Test the classifier on the given test set
    #     # Extract features from the test set and transform it using vectorizer
    #     test_text = []
    #     test_labels = []
    #     csv_reader = csv.reader(open(opts.test))
    #     for line in csv_reader:
    #         test_labels.append(int(line[0]))
    #         test_text.append(line[5])

    #     test_features = vectorizer.transform(test_text)

    #     # Print test mean accuracy
    #     print "Mean test accuracy:", classifier.score(test_features, test_labels)

    #     # Predict labels for the test set
    #     pred_test_labels = classifier.predict(test_features)

    #     # Print the classification report
    #     print classification_report(test_labels, pred_test_labels, target_names=["Negative Sentiment"," Positive Sentiment"])

    #     # Print the confusion matrix
    #     print confusion_matrix(test_labels, pred_test_labels)

    #     # Get predicted label of the test set
    #     if opts.classifier != 'svm':
    #         test_predicted_proba = classifier.predict_proba(test_features) # Use predict_proba
    #         util.plot_roc_curve(test_labels, test_predicted_proba) # Plot ROC curve

    #         # Print 10 correct and incorrect tweets with probabilities for writeup
    #         #print_examples(test_labels, pred_test_labels, test_predicted_proba, test_text)

    #     else:
    #         print "Confidence scores:", classifier.decision_function(test_features) # Use decision_function
    ############################################################

def print_top(num, vectorizer, classifier):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(classifier.classes_):
        top_n = numpy.argsort(classifier.coef_[i])[-num:]
        print "%s: %s" % (class_label, " ".join(feature_names[j] for j in top_n))

if __name__ == '__main__':
    main()
