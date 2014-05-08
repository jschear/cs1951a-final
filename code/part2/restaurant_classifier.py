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
from sklearn.metrics import *
from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
<<<<<<< HEAD
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
=======
from sklearn.svm import LinearSVC

>>>>>>> b117a0b917992c7c5b85d56162f5ee8e1a845495
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
<<<<<<< HEAD
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
=======

from tokenizer import Tokenizer

>>>>>>> b117a0b917992c7c5b85d56162f5ee8e1a845495
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
    parser.add_argument('-c', '--classifier', help='Classifier to use. Options are: RF, NB, LR, LOG')
    opts = parser.parse_args()
    ##

    ## Building training set
    # Initialize CountVectorizer
    tokenizer = Tokenizer(opts.stop)
    vectorizer = CountVectorizer(binary=True, lowercase=True, decode_error='replace', tokenizer=tokenizer)
    test_portion = .2


    # load business and categories
    all_categories = set(line.strip() for line in open(opts.categories))
    bids_to_categories = {}
    businesses_file = open(opts.businesses)
    for line in businesses_file:
        business = json.loads(line)
        categories = business['categories']
        if 'Restaurants' in categories:
            categories = filter(lambda x: x in all_categories, categories)
            if len(categories) > 0:
                bids_to_categories[business['business_id']] = categories
    print "Examining " + str(len(bids_to_categories)) + " Businesses"

    #pprint(bids_to_categories)

    # Load training review text
    print "-- Extracting Features --"
    reviews = []
    labels = []
    review_file = open(opts.reviews)
    i = 0
    for line in review_file:
        if opts.first != None and i > opts.first: break
        i+=1
        review = json.loads(line)
        # check if this is a review for one of the restuarants with labeled categories
        if review['business_id'] in bids_to_categories:
            reviews.append(review['text'])
            categories = bids_to_categories[review['business_id']]
            labels.append(categories)

    # shrink dataset
    if opts.first is not None:
        reviews = reviews[:opts.first]
        labels = labels[:opts.first]

    # count number of reviews for each label
    num_for_label = defaultdict(int)
    for label_list in labels:
        for label in label_list:
            num_for_label[label] = num_for_label[label] + 1

    num_for_label = sorted(num_for_label.items(), key=lambda x: x[1], reverse=True)
    print 'Number of reviews for each label:'
    print '\n'.join(map(lambda x: x[0] + ": " + str(x[1]), num_for_label))

    # Get training features using vectorizer
    assert len(reviews) == len(labels)
    splitindex = int((1-test_portion)*len(reviews))

    train_features = vectorizer.fit_transform(reviews[:splitindex])
    test_features = vectorizer.transform(reviews[splitindex:])

    # Transform training labels and test labels to numpy array (numpy.array)
    


    train_labels = numpy.array(labels[:splitindex])
    test_labels = numpy.array(labels[splitindex:])
    ############################################################

    # test_labels = numpy.array(test_labels)


    ##### TRAIN THE MODEL ######################################
    print "-- Training Classifier --"

    # try n_jobs = -1 for all of these once we get everything working
    if opts.classifier == 'RF':
        classifier = RandomForestClassifier(n_jobs = -1, verbose = 1, max_features = 'log2', max_depth = 5)
        train_features = train_features.toarray()
    elif opts.classifier == 'BNB':
        classifier = OneVsRestClassifier(BernoulliNB())
    elif opts.classifier == 'GNB':
        classifier = OneVsRestClassifier(GaussianNB())
        train_features = train_features.toarray()
    elif opts.classifier == 'SVC':
        classifier = OneVsRestClassifier(LinearSVC())
<<<<<<< HEAD
    elif opts.classifier == 'LR':
        classifier = OneVsRestClassifier(LogisticRegression())
    elif opts.classifier == 'KN':
        classifier = KNeighborsClassifier()
    elif opts.classifier == 'PPL':
        # classifier = Pipeline([('svm', LinearSVC()), ('lr', LogisticRegression())]) #takes forever
        classifier = Pipeline([('svm', LinearSVC()),('lr', OneVsRestClassifier(LogisticRegression()))])
    elif opts.classifier == '_LR':
        classifier = LogisticRegression()
    else:
=======
    elif opts.classifier == 'LOG':
        classifier = OneVsRestClassifier(LogisticRegression())
>>>>>>> b117a0b917992c7c5b85d56162f5ee8e1a845495
        print "Invalid classifier " + str(opts.classifier)
        return

    classifier.fit(train_features, train_labels)
    ############################################################


    ###### VALIDATE THE MODEL ##################################
    # Print training mean accuracy using 'score'
    print "-- Testing --"
    # print "Mean accuracy on training data:", classifier.score(train_features, train_labels)
<<<<<<< HEAD
    # pdb.set_trace()
    predicted_labels = classifier.predict(test_features)
    print classification_report(test_labels, predicted_labels)
    for evaluation_function in [accuracy_score, f1_score, lambda test_labels, predicted_labels : fbeta_score(test_labels, predicted_labels, .1), hamming_loss, jaccard_similarity_score, precision_score, recall_score, zero_one_loss]: 
        print evaluation_function.__name__ + ":" + str(evaluation_function(test_labels, predicted_labels))

    # for test_feature, label in zip(test_features, predicted_labels)[1:20]:
    #     # pdb.set_trace()
    #     print test_features, label
=======
    predicted_labels = classifier.predict(train_features)
    print accuracy_score(train_labels, predicted_labels)
    print classification_report(train_labels, predicted_labels)
>>>>>>> b117a0b917992c7c5b85d56162f5ee8e1a845495

    # TODO: Try the different metric here that are more interpretable for multilabel classification

    # Perform 5 fold cross validation (cross_validation.cross_val_score) with scoring='accuracy'
    # and print the mean score and std deviation
<<<<<<< HEAD
    # cv = 2

    # print "-- Cross-Validating with " + str(cv) + " folds -- "
    # scores = cross_validation.cross_val_score(classifier, train_features, train_labels,
    #     scoring='accuracy', cv = cv, n_jobs=cv) # passing integer for cv uses StratifiedKFold where k = integer
    # print scores, scores.mean()
=======
    #scores = cross_validation.cross_val_score(classifier, train_features, train_labels,
    #    scoring='accuracy', cv=5, n_jobs=-1)
>>>>>>> b117a0b917992c7c5b85d56162f5ee8e1a845495

    # print "Cross validation mean score:", numpy.mean(scores)
    # print "Cross validation standard deviation:", numpy.std(scores)
    ############################################################


    ##### EXAMINE THE MODEL ####################################
    print "-- Informative Features --"
    if opts.top is not None and opts.classifier != "RF":
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
#python restaurant_classifier.py -b ../../tmp/businesses.json -stop stopwords.txt -r ../../tmp/reviews.json -cats ./cuisines.txt -top 10 -c LR

def print_top(num, vectorizer, classifier):
    if type(classifier) == Pipeline:
        for name, classifier in classifier.named_steps.items():
            print name
            print_top(10, vectorizer, classifier)
    # pdb.set_trace()
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(classifier.classes_):
        top_n = numpy.argsort(classifier.coef_[i])[-num:]
        print "%s: %s" % (class_label, " ".join(feature_names[j] for j in top_n))

if __name__ == '__main__':
    main()
