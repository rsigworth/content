import demistomock as demisto
from CommonServerPython import *
from CommonServerUserPython import *
import pandas as pd
import json
from datetime import datetime
import base64
import sys
import dill
import numpy as np
import re

dill._dill._reverse_typemap['ClassType'] = type

import warnings

warnings.simplefilter("ignore")

########################################MODEL########################################################

import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import json
import pandas as pd
from scipy.spatial.distance import cdist

import pandas as pd
import json
from datetime import datetime
import base64
import sys
import dill
import numpy as np
import re

dill._dill._reverse_typemap['ClassType'] = type

import warnings

warnings.simplefilter("ignore")

########################################MODEL########################################################

import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import json
import pandas as pd
from scipy.spatial.distance import cdist


def check_list_of_dict(obj):
    return bool(obj) and all(isinstance(elem, dict) for elem in obj)


def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


regex_pattern = ["^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})Z", "(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2}).*"]


def recursive_filter(item, regex_patterns, *forbidden):
    if isinstance(item, list):
        return [recursive_filter(entry, regex_patterns, *forbidden) for entry in item if entry not in forbidden]
    if isinstance(item, dict):
        result = {}
        for key, value in item.items():
            value = recursive_filter(value, regex_patterns, *forbidden)
            if key not in forbidden and value not in forbidden and (not match_one_regex(value, regex_patterns)):
                result[key] = value
        return result
    return item


def match_one_regex(string, pattern):
    if not isinstance(string, str):
        return False
    if len(pattern) == 0:
        return False
    if len(pattern) == 1:
        return bool(re.match(pattern[0], string))
    else:
        return (match_one_regex(string, pattern[1:]) or bool(re.match(pattern[0], string)))


def normalize_json(obj):
    if isinstance(obj, float) or not obj:
        return " "
    if isinstance(obj, str):
        obj = json.loads(obj)
    if check_list_of_dict(obj):
        obj = {k: v for k, v in enumerate(obj)}
    if not isinstance(obj, dict):
        return " "
    my_dict = recursive_filter(obj, regex_pattern, "None", "N/A", None, "")
    my_string = json.dumps(my_dict)
    pattern = re.compile('([^\s\w]|_)+')
    my_string = pattern.sub(" ", my_string)
    my_string = my_string.lower()
    return my_string


def normalize_command_line(my_string):
    pattern_IP = r'(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])'
    if my_string and isinstance(my_string, str):
        my_string = my_string.lower()
        my_string = my_string.replace("Program Files (x86)", "ProgramFiles(x86)")
        my_string = my_string.replace("=", " = ")
        my_string = my_string.replace("\\", "/")
        my_string = my_string.replace("[", "")
        my_string = my_string.replace("]", "")
        # my_string = ' '.join([os.path.basename(x) for x in my_string.split(' ')])
        my_string = my_string.replace('"', "")
        my_string = my_string.replace("'", "")

        my_string = re.sub(pattern_IP, 'IP', my_string)
        return my_string
    else:
        return ''


def normalize_identity(my_string):
    if my_string and isinstance(my_string, str):
        return my_string
    else:
        return ''


def cdist_new(X, y):
    return np.maximum(1 - cdist(X, y)[:, 0], 0)  # , metric='cosine'


def identity(X, y):
    z = (X.to_numpy() == y.to_numpy()).astype(float)
    z[z == 0] = np.nan
    return z


class Tfidf(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, tfidf_params, normalize_function):
        self.feature_names = feature_names
        self.vec = TfidfVectorizer(**tfidf_params)
        self.normalize_function = normalize_function

    def fit(self, x, y=None):
        if self.normalize_function:
            x = x[self.feature_names].apply(self.normalize_function)
        self.vec.fit(x)
        return self

    def transform(self, x, y=None):
        if self.normalize_function:
            x = x[self.feature_names].apply(self.normalize_function)
        else:
            x = x[self.feature_names]
        return self.vec.transform(x).toarray()


class Identity(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, identity_params, normalize_function):
        self.feature_names = feature_names
        self.normalize_function = normalize_function

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        if self.normalize_function:
            return x[self.feature_names].apply(self.normalize_function)
        else:
            return x[self.feature_names]


TRANSFORMATION = {
    'commandline': {'transformer': Tfidf,
                    'normalize': normalize_command_line,
                    'params': {'analyzer': 'char', 'max_features': 1000, 'ngram_range': (1, 10)},
                    'scoring': {'scoring_function': cdist_new, 'min': 0.5}
                    },

    'url': {'transformer': Tfidf,
            'normalize': normalize_identity,
            'params': {'analyzer': 'char', 'max_features': 100, 'ngram_range': (1, 5)},
            'scoring': {'scoring_function': cdist_new, 'min': 0.5}
            },
    'potentialMatch': {'transformer': Identity,
                       'normalize': None,
                       'params': {},
                       'scoring': {'scoring_function': identity, 'min': 0.5}
                       },
    'json': {'transformer': Tfidf,
             'normalize': normalize_json,
             'params': {'analyzer': 'word', 'max_features': 5000, 'ngram_range': (1, 10), 'max_df': 0.2},
             'scoring': {'scoring_function': cdist_new, 'min': 0.5}
             }

}


class Transformer():
    def __init__(self, p_type, field, p_incidents_df, p_incident_to_match, p_params):
        self.type = p_type
        self.field = field
        self.incident_to_match = p_incident_to_match
        self.incidents_df = p_incidents_df
        self.params = p_params

    def fit_transform(self):
        transformation = self.params[self.type]
        transformer = transformation['transformer'](self.field, transformation['params'], transformation['normalize'])
        X_vect = transformer.fit_transform(self.incidents_df)
        incident_vect = transformer.transform(self.incident_to_match)
        return X_vect, incident_vect

    def get_score(self):
        scoring_function = self.params[self.type]['scoring']['scoring_function']
        X_vect, incident_vect = self.fit_transform()
        dist = scoring_function(X_vect, incident_vect)
        self.incidents_df['similarity %s' % self.field] = np.round(dist, 2)
        return self.incidents_df

    def filter(self):
        return self.incidents_df[self.incidents_df.dist < self.params[self.type]['scoring']['min']]


class Model:
    def __init__(self, p_transformation):
        self.transformation = p_transformation

    def init_prediction(self, p_incident_to_match, p_incidents_df, p_command_line=[], p_host_or_domain=[],
                        p_potential_exact_match=[], p_display_fields_incidents=[], p_json_fields=[]):
        self.incident_to_match = p_incident_to_match
        self.incidents_df = p_incidents_df
        self.command_line = p_command_line
        self.host_or_domain = p_host_or_domain
        self.potential_exact_match = p_potential_exact_match
        self.display_fields_incidents = p_display_fields_incidents
        self.json = p_json_fields

    def predict(self):
        self.remove_empty_field()
        self.get_score()
        self.compute_final_score()
        return self.display(), self.command_line + self.host_or_domain + self.potential_exact_match + self.json

    def remove_empty_field(self):
        remove_list = []
        for field in self.command_line:
            if not field in self.incident_to_match.columns or not self.incident_to_match[field].values[
                0] or not isinstance(self.incident_to_match[field].values[0], str) or \
                    self.incident_to_match[field].values[0] == 'None' or self.incident_to_match[field].values[
                0] == 'N/A':
                remove_list.append(field)
        self.command_line = [x for x in self.command_line if x not in remove_list]

        remove_list = []
        for field in self.host_or_domain:
            if not field in self.incident_to_match.columns or not self.incident_to_match[field].values[
                0] or not isinstance(self.incident_to_match[field].values[0], str) or \
                    self.incident_to_match[field].values[0] == 'None' or self.incident_to_match[field].values[
                0] == 'N/A':
                remove_list.append(field)
        self.host_or_domain = [x for x in self.host_or_domain if x not in remove_list]

        remove_list = []
        for field in self.potential_exact_match:
            if not field in self.incident_to_match.columns or not self.incident_to_match[field].values[
                0] or not isinstance(self.incident_to_match[field].values[0], str) or \
                    self.incident_to_match[field].values[0] == 'None' or self.incident_to_match[field].values[
                0] == 'N/A':
                remove_list.append(field)
        self.potential_exact_match = [x for x in self.potential_exact_match if x not in remove_list]

        remove_list = []
        for field in self.json:
            if not field in self.incident_to_match.columns or not self.incident_to_match[field].values[
                0] or self.incident_to_match[field].values[0] == 'None' or self.incident_to_match[field].values[
                0] == 'N/A' or all(not x for x in self.incident_to_match[field].values[0]):
                remove_list.append(field)
        self.json = [x for x in self.json if x not in remove_list]

    def get_score(self):
        for field in self.command_line:
            t = Transformer('commandline', field, self.incidents_df, self.incident_to_match, self.transformation)
            t.get_score()
        for field in self.host_or_domain:
            t = Transformer('url', field, self.incidents_df, self.incident_to_match, self.transformation)
            t.get_score()
        for field in self.potential_exact_match:
            t = Transformer('potentialMatch', field, self.incidents_df, self.incident_to_match, self.transformation)
            t.get_score()
        for field in self.json:
            t = Transformer('json', field, self.incidents_df, self.incident_to_match, self.transformation)
            t.get_score()

    def compute_final_score(self):
        col = self.incidents_df.loc[:,
              ['similarity %s' % field for field in self.command_line + self.host_or_domain + self.json]]
        self.incidents_df['final_score'] = np.round(col.mean(axis=1), 2)

    def display(self):
        self.compute_final_score()
        display_fields = remove_duplicates(
            self.display_fields_incidents + self.command_line + self.host_or_domain + self.potential_exact_match + [
                'similarity %s' % field for field in
                self.command_line + self.host_or_domain + self.json + self.potential_exact_match])
        df_sorted = self.incidents_df[display_fields + ['final_score']]  # .sort_values(by=display_fields)
        return df_sorted


########################################END MODEL####################################################

def display_similar_incident(similar_incidents, confidence, show_distance, max_incidents, fields_used, confidence_dict,
                             aggregate):
    similar_incidents['id'] = similar_incidents['id'].apply(lambda _id: "[%s](#/Details/%s)" % (_id, _id))

    if aggregate == 'True':
        agg_fields = [x for x in similar_incidents.columns if x not in ['id', 'created']]
        similar_incidents = similar_incidents.groupby(agg_fields, as_index=False, dropna=False).agg(
            {
                'created': lambda x: (min(x), max(x)) if len(x) > 1 else x,
                'id': lambda x: ' , '.join(x)}
        )

    similar_incidents.sort_values(by=['final_score'], ascending=False, inplace=True)
    if confidence:
        # similar_incidents = similar_incidents[similar_incidents.final_score < confidence_dict[confidence]] for high medium low
        similar_incidents = similar_incidents[similar_incidents.final_score >= confidence]
    if show_distance == 'False':
        col_to_remove = ['similarity %s' % field for field in fields_used]
        similar_incidents.drop(col_to_remove, axis=1, inplace=True)

    return similar_incidents.head(max_incidents)


# Get incident acording to incident id
def get_incidents_to_predict(incident_id, populate_fields):
    res = demisto.executeCommand('GetIncidentsByQuery', {
        'query': "id:(%s)" % incident_id,
        'populateFields': ' , '.join(populate_fields)
    })
    if is_error(res):
        return_error(res)
    if not json.loads(res[0]['Contents']):
        return_error("Incident with id:%s does not exists. Please check" % incident_id)
    else:
        incident = json.loads(res[0]['Contents'])
        return incident[0]


# Get incidents for a time window and exat match for somes fields
def get_all_incidents_for_time_window_and_exact_match(exact_match_fields, populate_fields, incident, from_date, to_date,
                                                      query_sup, limit):
    msg = ""
    exact_match_fields_list = []
    for i in exact_match_fields:
        exact_match_fields_list.append('%s: "%s"' % (i, incident[i]))
    query = " AND ".join(exact_match_fields_list)
    query += " AND -id:%s " % incident['id']
    if query_sup:
        query += " %s" % query_sup

    res = demisto.executeCommand('GetIncidentsByQuery', {
        'query': query,
        'populateFields': ' , '.join(populate_fields),
        'fromDate': from_date,
        'toDate': to_date,
        'limit': limit
    })
    if is_error(res):
        return_error(res)
    # sys.exit(0)
    incidents = json.loads(res[0]['Contents'])
    if len(incidents) == 0:
        return_error("No incident found with these exact match for the given date")
    if len(incidents) == limit:
        msg += "%s incidents fetched with exact match. Incident have been truncated due to query limit of %s. You will miss some incidents. Try to add exact matchs or increase limit argument" % (
            str(len(incidents)), str(limit))

    return incidents, msg


def get_prediction_for_incident():
    confidence_dict = {'HIGH': 0.2, 'MEDIUM': 0.4, 'LOW': 0.6, 'ALL': 100}
    # Create model and retrieve arguments
    model = Model(p_transformation=TRANSFORMATION)

    command_line_fields = demisto.args().get('fieldCommandLine', '').split(',')
    command_line_fields = [x.strip() for x in command_line_fields if x]

    host_or_domains_fields = demisto.args().get('fieldHostOrDomains', '').split(',')
    host_or_domains_fields = [x.strip() for x in host_or_domains_fields if x]

    potential_exact_match_fields = demisto.args().get('fieldPotentialExactMatch', '').split(',')
    potential_exact_match_fields = [x.strip() for x in potential_exact_match_fields if x]

    json_fields = demisto.args().get('fieldJson', '').split(',')
    json_fields = [x.strip() for x in json_fields if x]

    exact_match_fields = demisto.args().get('fieldExactMatch', '').split(',')
    exact_match_fields = [x.strip() for x in exact_match_fields if x]

    display_fields = demisto.args().get('fieldsToDisplay', '').split(',')
    display_fields = [x.strip() for x in display_fields if x]
    display_fields = list(set(['id', 'created', 'name'] + display_fields))

    from_date = demisto.args().get('fromDate')
    to_date = demisto.args().get('toDate')
    show_distance = demisto.args().get('showDistance')
    confidence = float(demisto.args().get('confidence'))
    max_incidents = int(demisto.args().get('maxIncidentsToDisplay'))
    query = demisto.args().get('query')
    aggregate = demisto.args().get('aggreagateIncidentsDifferentDate')
    limit = int(demisto.args()['limit'])

    global_msg = ""

    # load the Dcurrent incident
    incident_id = demisto.args().get('incidentId')
    populate_fields = command_line_fields + host_or_domains_fields + json_fields + potential_exact_match_fields + exact_match_fields + display_fields + [
        'id']
    if not incident_id:
        incident = demisto.incidents()[0]
        cf = incident.pop('CustomFields', {}) or {}
        incident.update(cf)
        incident = {k: v for k, v in incident.items() if k in populate_fields}
        incident_id = incident['id']
    else:
        incident = get_incidents_to_predict(incident_id, populate_fields)

    # load the related incidents
    populate_fields = display_fields + command_line_fields + host_or_domains_fields + json_fields + potential_exact_match_fields + exact_match_fields
    incidents, msg = get_all_incidents_for_time_window_and_exact_match(exact_match_fields, populate_fields, incident,
                                                                       from_date, to_date, query, limit)
    global_msg += "%s \n" % msg
    number_incident_fetched = len(incidents)

    incidents_df = pd.DataFrame(incidents)

    incorrect_fields = [i for i in populate_fields if i not in incidents_df.columns.tolist()]
    if incorrect_fields:
        global_msg += "%s \n" % "%s might not be correct spelling. Please correct or ignore this message" % ' , '.join(
            incorrect_fields)
        # return_error("%s is/are not correct fields. Please correct then or remove them"   %' , '.join(incorrect_fields))
    for fields in [display_fields, command_line_fields, host_or_domains_fields, json_fields,
                   potential_exact_match_fields]:
        for field in fields:
            if field not in incidents_df.columns.tolist():
                fields.remove(field)

    for field in incident.keys():
        if isinstance(incident[field], dict):
            incident[field] = json.dumps(incident[field])
    incident = pd.DataFrame.from_dict(incident, orient='index').T

    # incident.set_index('id', inplace=True)
    # demisto.results(str(command_line_fields))
    # sys.exit(0)

    # Model prediction
    model.init_prediction(incident, incidents_df, command_line_fields, host_or_domains_fields,
                          potential_exact_match_fields, display_fields, json_fields)
    similar_incidents, fields_used = model.predict()
    if len(fields_used) == 0:
        return_error(
            "No field are used to find similarity. No field selected OR Selected field are empty for this incident OR fields are misspelled OR field does not exist in the incidents fetched")

    similar_incidents = display_similar_incident(similar_incidents, confidence, show_distance, max_incidents,
                                                 fields_used, confidence_dict, aggregate)

    incidents_found = len(similar_incidents)

    incident_found_bool = (len(similar_incidents) > 0)

    # if not incident_found_bool:
    #     demisto.results("No incidents were found with the given threshold and the chosen fields")

    # Filter incident to investigate
    incident_filter = incident[[x for x in
                                display_fields + command_line_fields + host_or_domains_fields + json_fields + potential_exact_match_fields + exact_match_fields
                                if x in incident.columns]]

    # Convert dataframe output to JSON
    # similar_incidents_json = similar_incidents.to_dict(orient='rows')
    similar_incidents_json = similar_incidents.to_dict(orient='records')
    incident_json = incident_filter.to_dict(orient='records')

    # Format output
    # Format output
    col = similar_incidents.columns.tolist()
    col = ['id', 'created', 'name'] + [x for x in col if x not in ['id', 'created', 'name']]

    summary = {
        'Confidence': str(confidence),
        'Incident fetched with exact match': number_incident_fetched,
        'Number of similar incident found ': incidents_found,
        'Fields used for similarity (not empty)': ', '.join(fields_used),
        'Additional message': global_msg
    }

    if incident_found_bool:
        context = {
            'similarIncident': (similar_incidents[display_fields].to_dict(orient='records'))[0],
            # similar_incidents[display_fields].to_dict(orient='records'),
            'isSimilarIncidentFound': True
        }
    else:
        context = {
            'similarIncidentList': {},
            'isSimilarIncidentFound': False
        }

    return_outputs(readable_output=tableToMarkdown("Summary", summary))
    if incident_found_bool:
        # return_outputs(readable_output = tableToMarkdown("Actual incident", incident_json))
        return_outputs(readable_output=tableToMarkdown("Similar incidents", similar_incidents_json, col),
                       outputs={'DBotPredictSimilarEvents': context})


if __name__ in ['__main__', '__builtin__', 'builtins']:
    get_prediction_for_incident()
