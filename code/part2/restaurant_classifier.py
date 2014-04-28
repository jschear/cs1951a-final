from __future__ import division
import sys
import csv
import argparse
from collections import defaultdict
from operator import itemgetter

import util

import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import cross_validation
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from tokenizer import Tokenizer

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
    parser.add_argument('-reviews', required=True, help='Path to review data')
    parser.add_argument('-businesses', required=True, help='Path to business data')
    parser.add_argument('-c', '--classifier', default='nb', help='nb | log | svm')
    parser.add_argument('-top', type=int, help='Number of top features to show')
    opts = parser.parse_args()
    ##

    ## Building training set
    # Initialize CountVectorizerds
    tokenizer = Tokenizer()
    vectorizer = CountVectorizer(binary=True, lowercase=True, decode_error='replace', tokenizer=tokenizer)

    # load business
    businesses_file = open(opts.businesses)
    for line in businesses_file:
        business = json.loads(line)
        categories = business['categories']
        if 'Restaurants' in categories:
            categories.remove('Restaurants')
            if 'Price Range' in business['attributes']:

    # Load training review text
    review_file = open(opts.reviews)
    for line in review_file:
        review = json.loads(line)
        review['text']
        review['business_id']



    # Load training text and training labels
    train_text = []
    train_labels = []


    # Get training features using vectorizer
    train_features = vectorizer.fit_transform(train_text)

    # Transform training labels to numpy array (numpy.array)
    train_labels = numpy.array(train_labels)
    ############################################################


    ##### TRAIN THE MODEL ######################################
    # Initialize the corresponding type of the classifier and train it (using 'fit')
    if opts.classifier == 'nb':
        classifier = BernoulliNB(binarize=None)
    elif opts.classifier == 'log':
        classifier = LogisticRegression()
    elif opts.classifier == 'svm':
        classifier = LinearSVC()
    else:
        raise Exception('Unrecognized classifier!')

    classifier.fit(train_features, train_labels)
    ############################################################


    ###### VALIDATE THE MODEL ##################################
    # Print training mean accuracy using 'score'
    print "-- Validation --"
    print "Mean accuracy on training data:", classifier.score(train_features, train_labels)

    # Perform 10 fold cross validation (cross_validation.cross_val_score) with scoring='accuracy'
    # and print the mean score and std deviation
    scores = cross_validation.cross_val_score(classifier, train_features, train_labels,
        scoring='accuracy', cv=10, n_jobs=-1) # passing integer for cv uses StratifiedKFold where k = integer

    print "Cross validation mean score:", numpy.mean(scores)
    print "Cross validation standard deviation:", numpy.std(scores)
    ############################################################


    ##### EXAMINE THE MODEL ####################################
    if opts.top is not None:
        # print top n most informative features for positive and negative classes
        print "Top", opts.top, "most informative features:"
        util.print_most_informative_features(opts.classifier, vectorizer, classifier, opts.top)
    ############################################################


    ##### TEST THE MODEL #######################################
    print "-- Testing --"
    if opts.test is None:
        # Test the classifier on one sample test tweet
        # Tim Kraska 10:43 AM - 5 Feb 13
        test_tweet = 'Water dripping from 3rd to 1st floor while the firealarm makes it hard to hear anything. BTW this is the 2nd leakage.  Love our new house'

        sample_test_features = vectorizer.transform([test_tweet]) # extract features

        # Print the predicted label of the test tweet
        print "Test tweet predicted label:", classifier.predict(sample_test_features)[0]

        # Print the predicted probability of each label.
        if opts.classifier != 'svm':
            class_probs = classifier.predict_proba(sample_test_features)[0]
            print "Probability of label 0:", class_probs[0]
            print "Probability of label 1:", class_probs[1]
        else:
            print "Confidence score for test tweet:", classifier.decision_function(sample_test_features)[0]

    else:
        # Test the classifier on the given test set
        # Extract features from the test set and transform it using vectorizer
        test_text = []
        test_labels = []
        csv_reader = csv.reader(open(opts.test))
        for line in csv_reader:
            test_labels.append(int(line[0]))
            test_text.append(line[5])

        test_features = vectorizer.transform(test_text)

        # Print test mean accuracy
        print "Mean test accuracy:", classifier.score(test_features, test_labels)

        # Predict labels for the test set
        pred_test_labels = classifier.predict(test_features)

        # Print the classification report
        print classification_report(test_labels, pred_test_labels, target_names=["Negative Sentiment"," Positive Sentiment"])

        # Print the confusion matrix
        print confusion_matrix(test_labels, pred_test_labels)

        # Get predicted label of the test set
        if opts.classifier != 'svm':
            test_predicted_proba = classifier.predict_proba(test_features) # Use predict_proba
            util.plot_roc_curve(test_labels, test_predicted_proba) # Plot ROC curve

            # Print 10 correct and incorrect tweets with probabilities for writeup
            #print_examples(test_labels, pred_test_labels, test_predicted_proba, test_text)

        else:
            print "Confidence scores:", classifier.decision_function(test_features) # Use decision_function
    ############################################################

if __name__ == '__main__':
    main()
