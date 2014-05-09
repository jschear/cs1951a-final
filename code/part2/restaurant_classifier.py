from __future__ import division
import sys
import csv
import argparse
import json
from collections import defaultdict, Counter
from operator import itemgetter

from pprint import pprint

import util

from datetime import datetime

import pdb
import math
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import *
from sklearn import cross_validation
from itertools import chain

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


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
def time_generator():
    start_time = datetime.now()
    while(True):
        yield datetime.now() - start_time
timer = time_generator()


class VotingClassifier(object):

    def __init__(self, classifiers, weights = None):
        self.classifiers = classifiers
        if weights == None:
            weights = { classifier : 1.0 for classifier in self.classifiers }
        self.weights = weights
        self.num_classifiers = len(classifiers)

    def fit(self, features, labels):
        for classifier in self.classifiers:
            print '    -- Training ' + str(classifier.estimator_.__class__.__name__) + " -- " + str(timer.next())
            classifier.fit(features, labels)

    def predict(self, features):

        predictions = {}
        for classifier in self.classifiers:
            print '    -- Predicting with ' + str(classifier.__class__.__name__) + " -- " + str(timer.next())
            predictions[classifier] = classifier.predict(features)

        # predictions = { classifier: classifier.predict(features) for classifier in self.classifiers }
        out = []
        print '    -- Merging Predictions -- ' + str(timer.next())
        l = features.shape[0]
        for i, feature in enumerate(features):
            # print i, l
            if  i % (l//5) == 0:
                print "{0:.0f}".format((i/l)*100.0) + "%"
            # pdb.set_trace()
            feature_predictions = { classifier : predictions[classifier][i] for classifier in self.classifiers }
            out.append(self.merge(feature_predictions))
        return out

    def merge(self, predictions):
        classifiers, predictions = [classifier for classifier,_ in predictions.items()], [prediction for _,prediction in predictions.items()]
        filtered_predictions = filter(lambda prediction: len(prediction) > 0, predictions)
        out = []
        if len(filtered_predictions) == 0:
            # print str(filtered_predictions) + " ----> " + str(out)
            return out
        output_length = int(round(sum(map(len,filtered_predictions))/len(filtered_predictions)))
        if output_length == 0:
            output_length = 1
            # print str(filtered_predictions) + " ----> " + str(out)

        weighted = defaultdict(float)
        for classifier, prediction in zip(classifiers,predictions):
            for class_ in prediction:
                weighted[class_] += self.weights[classifier]

        out = [ class_ for class_,_ in sorted(weighted.items(), key = lambda item:item[1], reverse = True) ][:output_length]
        # print str(predictions) + " ----> " + str(out)
        return out
        # out =  [class_ for class_,_ in sorted(Counter(chain(*filtered_predictions)).items(),key = lambda item: item[1], reverse = True)]
        # print str(filtered_predictions) + " ----> " + str(out)
        # return out





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reviews', required=True, help='Path to review data')
    parser.add_argument('-b', '--businesses', required=True, help='Path to business data')
    parser.add_argument('-cats', '--categories', required=True, help='File of possible retaurant categories')
    parser.add_argument('-stop', required=True, help='Stopwords file')
    parser.add_argument('-top', type=int, help='Number of top features to show')
    parser.add_argument('-first', type=int, help='Number of reviews to use')
    parser.add_argument('-c', '--classifier', help='Classifier to use. Options are: RF, NB, LR, LOG')
    parser.add_argument('-o', '--output', help='Wordcloud output file dstination')
    opts = parser.parse_args()
    ##

    ## Building training set
    # Initialize CountVectorizer
    tokenizer = Tokenizer(opts.stop)
    vectorizer = CountVectorizer(binary=True, lowercase=True, decode_error='replace', tokenizer=tokenizer)
    test_portion = .2


    # load business and categories
    all_categories = set(line.strip() for line in open(opts.categories) if line[0] != "#")
    bids_to_categories = {}
    bids_to_names = {}
    businesses_file = open(opts.businesses)
    for line in businesses_file:
        business = json.loads(line)
        categories = business['categories']
        if 'Restaurants' in categories:
            categories = filter(lambda x: x in all_categories, categories)
            if len(categories) > 0:
                bids_to_categories[business['business_id']] = categories
                bids_to_names[business['business_id']] = business['name']
    print "Examining " + str(len(bids_to_categories)) + " Businesses"

    #pprint(bids_to_categories)

    # Load training review text
    print "-- Extracting Features -- " + str(timer.next())
    reviews = []
    labels = []
    review_file = open(opts.reviews)
    i = 0
    for line in review_file:
        if opts.first != None and i > opts.first: break
        i += 1
        review = json.loads(line)
        # check if this is a review for one of the restuarants with labeled categories
        if review['business_id'] in bids_to_categories:
            name = bids_to_names[review['business_id']]
            reviews.append(review['text'] + " " + name)
            categories = bids_to_categories[review['business_id']]
            labels.append(categories)

    # count number of reviews for each label
    num_for_label = defaultdict(int)
    for label_list in labels:
        for label in label_list:
            num_for_label[label] = num_for_label[label] + 1

    num_for_label = sorted(num_for_label.items(), key=lambda x: x[1], reverse=True)
    # print 'Number of reviews for each label:'
    # print '\n'.join(map(lambda x: x[0] + ": " + str(x[1]), num_for_label))

    # Get training features using vectorizer
    assert len(reviews) == len(labels)
    splitindex = int((1-test_portion)*len(reviews))

    train_features = vectorizer.fit_transform(reviews[:splitindex])
    test_features = vectorizer.transform(reviews[splitindex:])
    del reviews

    # Transform training labels and test labels to numpy array (numpy.array)
    train_labels = numpy.array(labels[:splitindex])
    test_labels = numpy.array(labels[splitindex:])
    del labels
    ############################################################

    # test_labels = numpy.array(test_labels)


    ##### TRAIN THE MODEL ######################################
    print "-- Training Classifier -- " + str(timer.next())

    # try n_jobs = -1 for all of these once we get everything working
    # some of these options don't really work...
    #add a voting algorithm using SVC, LogReg, and BNB? Weight them according to self calculated accuracy?
    if opts.classifier == 'RF':
        classifier = OneVsRestClassifier(RandomForestClassifier())
        train_features = train_features.toarray()
        test_features = test_features.toarray()

    elif opts.classifier == 'BNB':
        classifier = OneVsRestClassifier(BernoulliNB())
    elif opts.classifier == 'GNB':
        classifier = OneVsRestClassifier(GaussianNB())
        train_features = train_features.toarray()
        test_features = test_features.toarray()
    elif opts.classifier == 'SVC':
        classifier = OneVsRestClassifier(LinearSVC())
    elif opts.classifier == 'LR':
        params = { "intercept_scaling": 2.0, "dual": False, "C": 1.0, "dual" : False }
        classifier = OneVsRestClassifier(LogisticRegression(**params))
    elif opts.classifier == 'KN':
        classifier = KNeighborsClassifier()
    elif opts.classifier == 'PPL':
        # classifier = Pipeline([('svm', LinearSVC()), ('lr', LogisticRegression())]) #takes forever
        classifier = Pipeline([('svm', LinearSVC()),('lr', OneVsRestClassifier(LogisticRegression()))])
    elif opts.classifier == 'VOT':
        classifiers = [OneVsRestClassifier(LogisticRegression()), OneVsRestClassifier(BernoulliNB()), OneVsRestClassifier(LinearSVC())]
        weights = dict(zip(classifiers,[ 1.0, 0.8, 0.6 ]))
        classifier = VotingClassifier(classifiers, weights)
    else:
        print "Invalid classifier " + str(opts.classifier)
        return

    classifier.fit(train_features, train_labels)
    ############################################################


    ###### VALIDATE THE MODEL ##################################
    # Print training mean accuracy using 'score'
    print "-- Testing -- " + str(timer.next())
    # print "Mean accuracy on training data:", classifier.score(train_features, train_labels)
    predicted_labels = classifier.predict(test_features)
    print classification_report(test_labels, predicted_labels)
    # for evaluation_function in [accuracy_score, f1_score, lambda test_labels, predicted_labels : fbeta_score(test_labels, predicted_labels, .1), hamming_loss, jaccard_similarity_score, precision_score, recall_score, zero_one_loss]:
    #     print evaluation_function.__name__ + ":" + str(evaluation_function(test_labels, predicted_labels))

    print "Precision score"
    print precision_score(test_labels, predicted_labels, average = None)
    print precision_score(test_labels, predicted_labels, average = 'samples')

    print "Recall scores:"
    print recall_score(test_labels, predicted_labels, average = None)
    print recall_score(test_labels, predicted_labels, average = 'samples')

    print "f1 scores:"
    print f1_score(test_labels, predicted_labels, average = None)
    print f1_score(test_labels, predicted_labels, average = 'samples')

    print "Accuracy score:"
    print accuracy_score(test_labels, predicted_labels)

    print "Hamming loss:"
    print hamming_loss(test_labels, predicted_labels)

    print "Zero-one loss"
    print zero_one_loss(test_labels, predicted_labels)

    print "Jaccard similarity:"
    print jaccard_similarity_score(test_labels, predicted_labels)

    def evaluate(test_labels, predicted_labels):
        print classification_report(test_labels, predicted_labels)
        # pdb.set_trace()
        for evaluation_function in [accuracy_score, f1_score, lambda test_labels, predicted_labels : fbeta_score(test_labels, predicted_labels, .1), hamming_loss, jaccard_similarity_score, precision_score, recall_score, zero_one_loss]:
            print evaluation_function.__name__ + ":" + str(evaluation_function(test_labels, predicted_labels))

    evaluate(test_labels, predicted_labels)
    # for test_feature, label in zip(test_features, predicted_labels)[1:20]:
    #     # pdb.set_trace()
    #     print test_features, label

    # TODO: Try the different metric here that are more interpretable for multilabel classification

    # Perform 5 fold cross validation (cross_validation.cross_val_score) with scoring='accuracy'
    # and print the mean score and std deviation
    # cv = 2

    # print "-- Cross-Validating with " + str(cv) + " folds -- "
    # cv = 4
    # scores = cross_validation.cross_val_score(classifier, train_features, train_labels,
    #     scoring='accuracy', cv = cv, n_jobs=cv) # passing integer for cv uses StratifiedKFold where k = integer
    # print scores, scores.mean()

    #scores = cross_validation.cross_val_score(classifier, train_features, train_labels,
    #    scoring='accuracy', cv=5, n_jobs=-1)

    # print "Cross validation mean score:", numpy.mean(scores)
    # print "Cross validation standard deviation:", numpy.std(scores)
    ############################################################


    ##### EXAMINE THE MODEL ####################################
    if opts.top is not None and opts.classifier != "RF":
        print "-- Informative Features -- " + str(timer.next())
        # print top n most informative features for positive and negative classes
        print "Top", opts.top, "most informative features:"
        print_top(opts.top, vectorizer, classifier, opts.output)
    ############################################################


def print_top(num, vectorizer, classifier, output = None):
    # pdb.set_trace()
    try:
        classifier.coef_
    except Exception:
        print "Unable to print top " + str(num) + " results"
        return
    if type(classifier) == Pipeline:
        for name, classifier in classifier.named_steps.items():
            print name
            print_top(10, vectorizer, classifier)

    if type(classifier) == RandomForestClassifier:
        classifier = classifier.estimator

    if type(classifier) == VotingClassifier:
        for classifier_ in classifier.classifiers:
            print_top(num, vectorizer, classifier_)
        return
    # pdb.set_trace()
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    output_obj = {}
    for i, class_label in enumerate(classifier.classes_):
        top_n = numpy.argsort(classifier.coef_[i])[-num:]
        # for text, coefficient in zip(feature_names, classifier.coef_[i]):
            # pdb.set_trace()
        output_obj[class_label] = [{ "text" : text, "coefficient" : coefficient} for text, coefficient in sorted(zip(feature_names, list(classifier.coef_[i])), key = lambda item: item[1])][-20:]
        # output_obj[class_label]
        print "%s: %s" % (class_label, " ".join(feature_names[j] for j in top_n))
    with open(output,"w") as output_file:
        # pdb.set_trace()
        output_file.write(json.dumps(output_obj))

if __name__ == '__main__':
    main()
