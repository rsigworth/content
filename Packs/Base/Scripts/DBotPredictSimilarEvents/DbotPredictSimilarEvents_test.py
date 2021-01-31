from DBotPredictSimilarEvents import get_prediction_for_incident

from CommonServerPython import *


def executeCommand(command, args):
    with open('incidents.json', 'rb') as f:
        incidents = json.load(f)
    if command == 'GetIncidentsByQuery' and '-id' in args['query']:
        return [{'Contents': json.dumps(incidents), 'Type': 'note'}]
    else:
        return [{'Contents': json.dumps([incidents[0]]), 'Type': 'note'}]


def test_get_prediction_for_incident(mocker):
    mocker.patch.object(demisto, 'args',
                        return_value={
                            'incidentId': 12345,
                            'fieldCommandLine': 'entitycommandline, id',
                            'fieldPotentialExactMatch': '',
                            'fieldJson': '',
                            'limit': 10000,
                            'fieldExactMatch': '',
                            'fieldsToDisplay': '',
                            'showDistance': True,
                            'confidence': 0.2,
                            'maxIncidentsToDisplay': 100,
                            'query': '',
                            'aggreagateIncidentsDifferentDate': 'True',
                        })

    mocker.patch.object(demisto, 'executeCommand', side_effect=executeCommand)
    res = get_prediction_for_incident()
    print(res)
