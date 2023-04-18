import json
import pickle
from collections import defaultdict
from graph_generator import schema_generator

with open('2021-12-18.pkl', 'rb') as f:
    data = pickle.load(f)
filename = 'output/Bombing Attacks.txt'
with open(filename) as f:
    lines = f.readlines()

def DFS(G, v, seen=None, path=None):
    if seen is None: seen = []
    if path is None: path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(DFS(G, t, seen[:], t_path))
    return paths


def construct_longest_time_chains(temporal_rel):
    G = defaultdict(list)
    sources = []
    targets = []
    for (s, t) in temporal_rel:
        G[s].append(t)
        sources.append(s)
        targets.append(t)

    real_sources = [x for x in sources if not x in targets]
    max_paths = []
    for v in real_sources:
        all_paths = DFS(G, v)
        max_len = max(len(p) for p in all_paths)
        max_paths_v = [p for p in all_paths if len(p) == max_len]
        max_paths.extend(max_paths_v)
    return max_paths


def find_context(topic, document_num, line_num, index):
    rel_data = data[topic]
    Not_found = False
    if len(rel_data) <= int(document_num):
        Not_found = True
        return False, False, True
    else:
        rel_document = rel_data[int(document_num)]
    if len(rel_document.split('\n')) <= int(line_num):
        Not_found = True
        return False, False, True
    else:
        rel_line = rel_document.split('\n')[int(line_num)]
    if len(rel_line.split(' ')) <= int(index):
        Not_found = True
        return False, False, True
    else:
        mention = rel_line.split(' ')[int(index)]

    return rel_line, mention, Not_found


def read_argument_mentions(filename, events_starts, event_name):
    arg_mentions = {'ARG0': [], 'ARG1': []}
    # with open(filename) as f:
    #     lines = f.readlines()
    for i in range(0, events_starts):
        rel_string = lines[i]
        tab_split = rel_string.split('\t')
        if rel_string.startswith('Event: \'' + event_name):
            for tab in tab_split:
                if tab.startswith('ARG0:'):
                    NER_type = tab.split(',')[0].split('(')[1]
                    if NER_type != 'NA':
                        mention = tab.split(',')[1].removesuffix(')')
                        arg_mentions['ARG0'].append(mention)
                if tab.startswith('ARG1:'):
                    NER_type = tab.split('(')[1].split(',')[0]
                    if NER_type != 'NA':
                        mention = tab.split(',')[1].removesuffix(')')
                        arg_mentions['ARG1'].append(mention)
    return arg_mentions


def read_mentions(filename):
    # with open(filename) as f:
    #     lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith('top 30 events:'):
            events_start = i + 2
        if line.startswith('top 30 temporal relations:'):
            temporal_starts = i + 2
            break
    events_ends = temporal_starts - 3
    verbs = {}
    for i in range(events_start, events_ends):
        rel_string = lines[i]
        event = rel_string.split('\'')[1]
        verbs[event] = {}
        event_mentions_str = rel_string.split('{')[1]
        event_mentions_str = event_mentions_str.removesuffix('}, arguments: ')
        verbs[event]['event_mentions'] = event_mentions_str.split(', ')
        verbs[event]['arg_mentions'] = read_argument_mentions(filename, events_start, event)
    # print(verbs[event]['event_mentions'])
    return verbs


def convert_2_json(schema, filename):
    schema_json = {}
    schema_json['@context'] = ['', {'caci': '', 'my_key': '', 'giant_bitsring': ''}]
    schema_json['@id'] = ''
    schema_json['sdfVersion'] = ''
    schema_json['version'] = ''
    schema_json['ta2'] = False
    schema_json['comment'] = ['zero shot schema', 'this schema was induced using non-real documents (generated from GPT3)']
    schema_json['events'] = []
    schema_json['entities'] = []
    schema_json['relations'] = []
    verbs = read_mentions(filename)
    topic = schema['topic'].split('/')[1]
    if 'direct' in topic:
        return None
    # First event is the schema topic
    topic_event = {'@id': '', 'name': topic, 'description': 'schema topic', 'goal': '', 'privateData': {},
                   'children_gate': 'and', 'children': []}
    schema_json['events'].append(topic_event)
    temporal_chains = construct_longest_time_chains(schema['temporal'])
    hierarchical_chains = construct_longest_time_chains(schema['subevent'])
    event_list_file = list(verbs.keys())
    event_list_schema = []
    for item in schema['pred_and_args']:
        event_list_schema.append(list(item.keys())[0])
    entity_counter_id = 0
    event_counter_id = 0
    participants_counter_id = 0
    for event in event_list_schema:
        if event in event_list_file:
            # check if has children
            has_children = False
            has_parent = False
            for rel in schema['subevent']:
                if event == rel[0]:
                    has_children = True
                    break
                if event == rel[1]:
                    has_parent = True

            mentions = verbs[event]['event_mentions']
            if has_children:
                event_dict = {'@id': 'Events: ' + str(event_counter_id), 'name': event, 'ta1explanation': '',
                              'qnode': '', 'qlabel': '', 'participants': [], 'children_gate': 'and', 'children': []}
            else:
                event_dict = {'@id': 'Events: ' + str(event_counter_id), 'name': event, 'ta1explanation': '',
                              'qnode': '', 'qlabel': '', 'participants': []}
            event_counter_id += 1

            # here description is the context of the event, i.e., the sentence it was mentioned in
            # (later extract qnode and qlabel from this context)
            Not_found = True
            j = 0
            while Not_found and j < len(mentions):
                if mentions[j].endswith('}'):
                    mentions[j] = mentions[j].removesuffix('}')

                first_mention = mentions[j].split('_')
                # print(first_mention)
                event_dict['description'], event_name, Not_found = \
                    find_context(topic, first_mention[0].removeprefix('\''),
                                 first_mention[1], first_mention[2].removesuffix('\''))
                j += 1

            #---------------------------------------- Entities ----------------------------------------#
            # add participants to event dict and entities to entities list is the json schema
            arg0_mentions = verbs[event]['arg_mentions']['ARG0']
            arg1_mentions = verbs[event]['arg_mentions']['ARG1']
            if arg0_mentions:
                entity_dict = {'@id': 'Entities: ' + str(entity_counter_id), 'entity': '',
                                    'qnode': '', 'qlabel': '', 'centrality': 1}
                entity_counter_id += 1
                participant_dict = {'@id': 'Participant: ' + str(participants_counter_id), 'roleName': 'arg0',
                                    'entity': entity_dict['@id']}
                participants_counter_id += 1

                Not_found = True
                j = 0
                while Not_found:
                    first_mention = arg0_mentions[0].split('_')
                    entity_dict['source_sentence'], entity_dict['name'], Not_found \
                        = find_context(topic, first_mention[0].removeprefix('\''),
                                       first_mention[1], first_mention[2].removesuffix('\''))
                    j += 1

                event_dict['participants'].append(participant_dict)
                schema_json['entities'].append(entity_dict)

            if arg1_mentions:
                entity_dict = {'@id': 'Entities: ' + str(entity_counter_id), 'entity': '',
                               'qnode': '', 'qlabel': '', 'centrality': 1}
                entity_counter_id += 1
                participant_dict = {'@id': 'Participant: ' + str(participants_counter_id), 'roleName': 'arg1',
                                    'entity': entity_dict['@id']}
                participants_counter_id += 1

                Not_found = True
                j = 0
                while Not_found:
                    first_mention = arg1_mentions[0].split('_')
                    entity_dict['source_sentence'], entity_dict['name'], Not_found = \
                        find_context(topic, first_mention[0].removeprefix('\''),
                                     first_mention[1], first_mention[2].removesuffix('\''))
                    j += 1

                event_dict['participants'].append(participant_dict)
                schema_json['entities'].append(entity_dict)

            # ---------------------------------------- Children ----------------------------------------#
            if not has_parent:  # should be added to the list of children of topic event
                child_dict = {'child': event_dict['@id'], 'outlinks': []}
                schema_json['events'][0]['children'].append(child_dict)
                for temporal_chain in temporal_chains:
                    if event in temporal_chain:
                        index = temporal_chain.index(event)
                        if index != len(temporal_chain) - 1:
                            child_dict['outlinks'].append(temporal_chain[index + 1])  # TODO fix later for id instead of verb

            # add its children and the relations between them
            if has_children:
                for hierarchical_chain in hierarchical_chains:
                    if event in hierarchical_chain:
                        index = hierarchical_chain.index(event)
                        if index != len(hierarchical_chain) - 1:
                            child_dict = {'child': hierarchical_chain[index + 1], 'outlinks': []}
                            event_dict['children'].append(child_dict)
                            for temporal_chain in temporal_chains:
                                if hierarchical_chain[index + 1] in temporal_chain:
                                    t_index = temporal_chain.index(hierarchical_chain[index + 1])
                                    if t_index != len(temporal_chain) - 1:
                                        child_dict['outlinks'].append(temporal_chain[t_index + 1])
                                    # TODO fix later for id instead of verb

            schema_json['events'].append(event_dict)


    return schema_json


if __name__ == '__main__':

    schema = schema_generator(filename)
    if 'direct' in filename.split('/')[0]:
        schema['topic'] = filename.split('.')[0] + '_direct'
    else:
        schema['topic'] = filename.split('.')[0]
    schema_json = convert_2_json(schema, filename)
    print(json.dumps(schema_json))