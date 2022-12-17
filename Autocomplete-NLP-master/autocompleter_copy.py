import json
import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances


class LoadingData():
    def __init__(self):
        pass

    def load_data(self):
        train_file_path = os.path.join("Train")
        validation_file_path = os.path.join("Validate")
        category_id = 0
        self.cat_to_intent = {}
        self.intent_to_cat = {}

        for dirname, _, filenames in os.walk(train_file_path):
            for filename in filenames:
                file_path = os.path.join(dirname, filename)
                intent_id = filename.replace(".json", "")
                self.cat_to_intent[category_id] = intent_id
                self.intent_to_cat[intent_id] = category_id
                category_id += 1
        print(self.cat_to_intent)
        print(self.intent_to_cat)
        '''Training data'''
        training_data = list()
        for dirname, _, filenames in os.walk(train_file_path):
            for filename in filenames:
                file_path = os.path.join(dirname, filename)
                intent_id = filename.replace(".json", "")
                training_data += self.make_data_for_intent_from_json(
                    file_path, intent_id, self.intent_to_cat[intent_id])
        self.train_data_frame = pd.DataFrame(
            training_data, columns=['Text', 'intent', 'index'])

        self.train_data_frame = self.train_data_frame.sample(frac=1)

        '''Validation data'''
        validation_data = list()
        for dirname, _, filenames in os.walk(validation_file_path):
            for filename in filenames:
                file_path = os.path.join(dirname, filename)
                intent_id = filename.replace(".json", "")
                validation_data += self.make_data_for_intent_from_json(
                    file_path, intent_id, self.intent_to_cat[intent_id])
        self.validation_data_frame = pd.DataFrame(
            validation_data, columns=['Text', 'intent', 'index'])

        self.validation_data_frame = self.validation_data_frame.sample(frac=1)
        return self.train_data_frame

    def make_data_for_intent_from_json(self, json_file, intent_id, cat):
        json_d = json.load(open(json_file))

        json_dict = json_d[intent_id]

        sent_list = list()
        for i in json_dict:
            each_list = i['data']
            sent = ""
            for i in each_list:
                sent = sent + i['text'] + " "
            sent = sent[:-1]
            for i in range(3):
                sent = sent.replace("  ", " ")
            sent_list.append((sent, intent_id, cat))
        return sent_list


def splitDataFrameList(df, target_column, separator):

    def split_text(line, separator):
        splited_line = [e+df for e in line.split(separator) if e]
        return splited_line

    def splitListToRows(row, row_accumulator, target_column, separator):
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(splitListToRows, axis=1, args=(
        new_rows, target_column, separator))
    new_df = pd.DataFrame(new_rows)
    return new_df


class Autocompleter:
    def __init__(self):
        pass

    def process_data(self, new_df):

        print("split sentenses on punctuation...")
        for sep in ['. ', ', ', '? ', '! ', '; ']:
            new_df = splitDataFrameList(new_df, 'Text', sep)

        print("Text Cleaning using simple regex...")
        new_df['Text'] = new_df['Text'].apply(lambda x: " ".join(x.split()))
        new_df['Text'] = new_df['Text'].apply(lambda x: x.strip("."))
        new_df['Text'] = new_df['Text'].apply(lambda x: " ".join(x.split()))
        new_df['Text'] = new_df['Text'].apply(
            lambda x: x.replace(' i ', ' I '))
        new_df['Text'] = new_df['Text'].apply(lambda x: x.replace(' ?', '?'))
        new_df['Text'] = new_df['Text'].apply(lambda x: x.replace(' !', '!'))
        new_df['Text'] = new_df['Text'].apply(lambda x: x.replace(' .', '.'))
        new_df['Text'] = new_df['Text'].apply(lambda x: x.replace('OK', 'Ok'))
        # new_df['Text'] = new_df['Text'].apply(lambda x: x[0].upper()+x[1:])
        new_df['Text'] = new_df['Text'].apply(
            lambda x: x+"?" if re.search(r'^(Wh|How).+([^?])$', x) else x)

        print("calculate nb words of sentenses...")
        new_df['nb_words'] = new_df['Text'].apply(
            lambda x: len(str(x).split(' ')))
        new_df = new_df[new_df['nb_words'] > 2]

        print("count occurence of sentenses...")
        new_df['Counts'] = new_df.groupby(['Text'])['Text'].transform('count')

        print("remove duplicates (keep last)...")
        new_df = new_df.drop_duplicates(subset=['Text'], keep='last')

        new_df = new_df.reset_index(drop=True)
        print(new_df.shape)

        return new_df

    def calc_matrice(self, df):
        # define tfidf parameter in order to count/vectorize the description vector and then normalize it.
        model_tf = TfidfVectorizer(
            analyzer='word', ngram_range=(1, 5), min_df=0)
        tfidf_matrice = model_tf.fit_transform(df['Text'])
        print("tfidf_matrice ", tfidf_matrice.shape)
        return model_tf, tfidf_matrice

    def generate_completions(self, prefix_string, data, model_tf, tfidf_matrice):

        prefix_string = str(prefix_string)
        new_df = data.reset_index(drop=True)
        weights = new_df['Counts'].apply(lambda x: 1 + np.log1p(x)).values

        # tranform the string using the tfidf model
        tfidf_matrice_spelling = model_tf.transform([prefix_string])
        # calculate cosine_matrix
        cosine_similarite = linear_kernel(
            tfidf_matrice, tfidf_matrice_spelling)

        # sort by order of similarity from 1 to 0:
        similarity_scores = list(enumerate(cosine_similarite))
        similarity_scores = sorted(
            similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[0:10]

        similarity_scores = [i for i in similarity_scores]
        similarity_indices = [i[0] for i in similarity_scores]

        # add weight to the potential results that had high frequency in orig data
        for i in range(len(similarity_scores)):
            similarity_scores[i][1][0] = similarity_scores[i][1][0] * \
                weights[similarity_indices][i]

        similarity_scores = sorted(
            similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[0:3]
        similarity_indices_w = [i[0] for i in similarity_scores]

        return new_df.loc[similarity_indices_w]['Text'].tolist()
