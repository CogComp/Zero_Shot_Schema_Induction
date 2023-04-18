import re
from collections import Counter
import json
import os


def fix_timelines(temporal_rel):
    # 1. create complete timelines (A,B), (B,C), (A,C), (B,D), (C,D) => (A,D) and of other lengths of timelines
    list_of_time_verbs = set([item for timeline in temporal_rel for item in timeline])
    temporal_rel_dict = {verb: [] for verb in list_of_time_verbs}
    for temp_r in temporal_rel:
        temporal_rel_dict[temp_r[0]].append(temp_r[1])

    sorted_temp_rel_dict = sorted(temporal_rel_dict, key=lambda k: len(temporal_rel_dict[k]), reverse=True)
    # print(sorted_temp_rel_dict)
    # sort every list of children according to the length of their list of children
    for key in sorted_temp_rel_dict:
        temporal_rel_dict[key].sort(key=lambda k: sorted_temp_rel_dict.index(k))

    timelines = create_timelines(temporal_rel_dict)
    # add missing couples according to timelines
    for timeline in timelines:
        for ind, a in enumerate(timeline):
            for j in range(ind + 1, len(timeline)):
                b = timeline[j]
                if not (a, b) in temporal_rel:
                    temporal_rel.append((a, b))
    return temporal_rel


def create_timelines(temporal_dict):
    def adjlist_find_paths(a, n, m, path=[]):
        "Find paths from node index n to m using adjacency list a."
        path = path + [n]
        if n == m:
            return [path]
        paths = []
        for child in a[n]:
            if child not in path:
                child_paths = adjlist_find_paths(a, child, m, path)
                for child_path in child_paths:
                    paths.append(child_path)
        return paths

    destinations = [x for x in temporal_dict.keys() if not temporal_dict[x]]
    sources = [x for x in temporal_dict.keys() if temporal_dict[x]]

    final_paths = []
    for n in sources:
        for m in destinations:
            final_paths.extend(adjlist_find_paths(temporal_dict, n, m))
    # filter our paths that are included in other paths:
    # print(final_paths)
    copy_paths = final_paths.copy()
    for path in copy_paths:
        for path2 in copy_paths:
            if path == path2 or not path in final_paths:
                continue
            elif set(path) <= set(path2):
                ind = final_paths.index(path)
                final_paths.pop(ind)

    return final_paths


def fix_schema(schema):
    temporal_rel = schema['temporal']
    subevent_rel = schema['subevent']
    or_rel = schema['OR']
    and_rel = schema['AND']

    temporal_rel = fix_timelines(temporal_rel)

    # 2. (A OR B) + (B AND C) => (A OR C)
    copy_or_rel = or_rel.copy()
    for and_r in and_rel:
        for i in range(len(and_r)):
            B = and_r[i]
            if i == 0:
                other_i = 1
            else:
                other_i = 0
            C = and_r[other_i]
            for or_r in copy_or_rel:
                if and_r[i] in or_r:
                    ind = or_r.index(B)
                    if ind == 0:
                        other_ind = 1
                    else:
                        other_ind = 0
                    A = or_r[other_ind]
                    if not (A, C) in or_rel or not (C, A) in or_rel:
                        or_rel.append((A, C))

    # 3. complete timelines for AND relation: if (A AND B) + (A,C) => (B,C)
    copy_temp_rel = temporal_rel.copy()
    for and_r in and_rel:
        for i in range(len(and_r)):
            A = and_r[i]
            if i == 0:
                other_i = 1
            else:
                other_i = 0
            B = and_r[other_i]
            for temp_r in copy_temp_rel:
                if A in temp_r:
                    ind = temp_r.index(A)
                    if ind == 0:
                        other_ind = 1
                    else:
                        other_ind = 0
                    C = temp_r[other_ind]
                    if ind == 0 and not (B, C) in temporal_rel:
                        temporal_rel.append((B, C))
                    if ind == 1 and not (C, B) in temporal_rel:
                        temporal_rel.append((C, B))

    # 4. If (A OR B) and (B OR A) => omit (B OR A)
    or_rel_c = []
    for x in or_rel:
        if x not in or_rel_c:
            if (x[1], x[0]) not in or_rel_c:
                or_rel_c.append(x)
    and_rel_c = []
    for x in and_rel:
        if x not in and_rel_c:
            if (x[1], x[0]) not in and_rel_c:
                and_rel_c.append(x)

    or_rel = or_rel_c
    and_rel = and_rel_c

    # 5. delete subevent that clash with temporal after the transitivity resolve
    subevent_rel_c = []
    for item in subevent_rel:
        if not item in temporal_rel and not (item[1], item[0]) in temporal_rel:
            subevent_rel_c.append(item)
    subevent_rel = subevent_rel_c

    return schema


def resolve_transitivity(relation):
    symmetric_relation = []
    for rel in relation:
        symmetric_relation.append(rel)
        symmetric_relation.append((rel[1], rel[0]))

    transitive_relation = []
    for rel in symmetric_relation:
        for rel2 in symmetric_relation:
            if rel == rel2 or rel == (rel2[1], rel2[0]):
                continue
            else:
                if rel[1] == rel2[0] and not (rel2[1], rel[0]) in transitive_relation:
                    transitive_relation.append((rel[0], rel2[1]))

    partial_order_relation = set(relation).union(transitive_relation)

    return partial_order_relation


def check_4_support_noun_verbs(file_name, relation, list_of_docs):
    with open(file_name) as f:
        lines = f.readlines()
    for verb in relation:
        support = []
        for line in lines:
            if line.startswith('Event: \'' + verb):
                rel_doc = line.split('(')[1].split('_')[0]
                if rel_doc in list_of_docs:
                    if 'Support: ' in line:
                        support.append(True)
                    else:
                        support.append(False)
                        break
        if all(ele for ele in support):
            return True

    return False


def build_hierarchy(subevents, timelines):
    list_of_time_verbs = set([item for timeline in timelines for item in timeline])

    flat_list = [item for sublist in subevents.values() for item in sublist]
    frequent_subevents = Counter(flat_list)
    subevent_relations_c = set(flat_list)
    # delete contradicting subevent relations (A,B) and (B,A)
    subevent_relations = []
    for item in subevent_relations_c:
        if (item[1], item[0]) in subevent_relations_c:
            continue
        else:
            subevent_relations.append(item)

    filtered_subevent = []
    for item in subevent_relations:
        if (item[0] in list_of_time_verbs and item[1] in list_of_time_verbs) or frequent_subevents[item] > 2: #TODO: maybe increase?
            filtered_subevent.append(item)

    return filtered_subevent
    # parents = list(set(parents))
    # children = list(set(children))

    # #change timelines according to hierarchy: split timelines that involve different hierarchies
    # # if two events in a timeline are parent and child, consider only the parent (parent, child, other event) -> (parent, other event)
    # # if three events in a timeline are parent and two children, consider only the children (parent, child1, child2) -> (child1, child2)
    # children_timelines = []
    # parent_timelines = []
    # for timeline in timelines:
    #     flag_parents = False
    #     flag_children = False
    #     # check if all verbs in the timeline are on the same hierarchy:
    #     for verb in timeline:
    #         if not verb in children and not verb in parents:  # if an event doesn't have children then it's in the highest hierarchy
    #             parents.append(verb)
    #         if verb in children and verb in parents:  # judge according to the highest hierarchy
    #             flag_parents = True
    #         if verb in children and not verb in parents:
    #             flag_children = True
    #         if verb in parents and not verb in children:
    #             flag_parents = True
    #
    #     if flag_children and not flag_parents:
    #         children_timelines.append(timeline)
    #     if flag_parents and not flag_children: # need to check for inconsistensies and if there is a hierarchy
    #         parent_timelines.append(timeline)
    #     if flag_parents and flag_children:
    #         timelinep = []
    #         timelinec = []
    #         for verb in timeline:
    #             if verb in parents:
    #                 timelinep.append(verb)
    #             else:
    #                 timelinec.append(verb)
    #         if len(timelinec) > 1:
    #             children_timelines.append(tuple(timelinec))
    #         if len(timelinep) > 1:
    #             parent_timelines.append(tuple(timelinep))
    #
    # print('children timelines: ', children_timelines)
    # print('parents timelines: ', parent_timelines)


def build_schema(timelines, subevents, corefs):
    subevent_dict = {}
    for item in subevents:
        if item[0] in subevent_dict.keys():
            subevent_dict[item[0]].append(item[1])
        else:
            subevent_dict[item[0]] = [item[1]]

    subevent_chains = construct_longest_chains(subevent_dict)

    verb_hierarchy_ratings = {}
    for chain in subevent_chains:
        for i, verb in enumerate(chain):
            if verb in verb_hierarchy_ratings.keys():
                prior_value = verb_hierarchy_ratings[verb]
                verb_hierarchy_ratings[verb] = max(prior_value, i)
            else:
                verb_hierarchy_ratings[verb] = i

    # set the level of hierarchy for every verb in timelines
    for timeline in timelines:
        for i, verb in enumerate(timeline):
            if verb in verb_hierarchy_ratings.keys():
                continue
            else:
                if i == 0:
                    for item in timeline:
                        # set the hierarchy based on the closest verb in the timeline that does have a level
                        if item in verb_hierarchy_ratings.keys():
                            verb_hierarchy_ratings[verb] = verb_hierarchy_ratings[item]
                    # if no verb in the timeline has a level then set them all to level 0
                    if not verb in verb_hierarchy_ratings.keys():
                        verb_hierarchy_ratings[verb] = 0
                else:
                    verb_hierarchy_ratings[verb] = verb_hierarchy_ratings[timeline[i-1]]
    # print(verb_hierarchy_ratings)
    relevant_subevent_relations = []
    schema_time_relations = []
    for timeline in timelines:
        for i, e1 in enumerate(timeline):
            for j in range(i+1, len(timeline)):
                e2 = timeline[j]
                level_e1 = verb_hierarchy_ratings[e1]
                level_e2 = verb_hierarchy_ratings[e2]
                if (e1, e2) in subevents:
                    relevant_subevent_relations.append((e1, e2))
                elif (e2, e1) in subevents:
                    relevant_subevent_relations.append((e2, e1))
                elif level_e1 == level_e2: # only if the verbs are on the same level of hierarchy we consider the
                    # temporal relation
                    schema_time_relations.append((e1, e2))
                else:
                    continue

    relevant_subevent_relations = sorted(list(set(relevant_subevent_relations).union(subevents)))
    schema_time_relations = sorted(list(set(schema_time_relations)))
    schema_time_relations = fix_timelines(schema_time_relations)
    # print(schema_time_relations)
    # logical relations: currently without taking arguments into account
    AND_rel = []
    OR_rel = []
    for rel1 in schema_time_relations:
        for rel2 in schema_time_relations:
            if rel1 == rel2:
                continue
            else:
                if rel1[0] == rel2[0]:
                    if not (rel2[1], rel1[1]) in OR_rel and not (rel1[1], rel2[1]) in schema_time_relations \
                            and not (rel2[1], rel1[1]) in schema_time_relations:
                        OR_rel.append((rel1[1], rel2[1]))
                if rel1[1] == rel2[1]:
                    if not (rel2[0], rel1[0]) in OR_rel and not (rel1[0], rel2[0]) in schema_time_relations \
                            and not (rel2[0], rel1[0]) in schema_time_relations:
                        OR_rel.append((rel1[0], rel2[0]))
                if rel1[1] == rel2[0] and rel1[0] == rel2[1]:
                    if (rel1[0], rel1[1]) in AND_rel or (rel1[1], rel1[0]) in AND_rel:
                        continue
                    else:
                        AND_rel.append((rel1[0], rel1[1]))

    # adjust temporal relations according to AND and OR relations - delete from temporal every relation that appears
    # in logical and complete logical to make it transitive
    # starting with making logical relation transitive
    AND_REL = list(resolve_transitivity(AND_rel))
    OR_REL = list(resolve_transitivity(OR_rel))
    # if a relation in temporal appears in AND_REL, pop from temporal:
    temporal_REL = []
    for item in schema_time_relations:
        if not item in AND_REL and not (item[1], item[0]) in AND_REL:
            temporal_REL.append(item)
    # if a couple appears in AND and in OR, then it should be only in AND:
    for item in AND_REL:
        if item in OR_REL:
            OR_REL.pop(OR_REL.index(item))
        if (item[1], item[0]) in OR_REL:
            OR_REL.pop(OR_REL.index((item[1], item[0])))
    # if a couple appears in OR and in temporal, then it should be only in temporal:
    for item in temporal_REL:
        if item in OR_REL:
            OR_REL.pop(OR_REL.index(item))
        if (item[1], item[0]) in OR_REL:
            OR_REL.pop(OR_REL.index((item[1], item[0])))

    subevent_REL = []
    for item in relevant_subevent_relations:
        if not item in AND_REL and not (item[1], item[0]) in AND_REL:
            subevent_REL.append(item)

    schema = {'temporal': temporal_REL, 'subevent': subevent_REL, 'OR': OR_REL, 'AND': AND_REL}

    return schema


# old code
# def build_schema(timelines, subevents, corefs):
#     # build par:children dict for subevent relations:
#     child_par_dict = {}
#     for relation in subevents:
#         if relation[1] in child_par_dict.keys():
#             child_par_dict[relation[1]].append(relation[0])
#         else:
#             child_par_dict[relation[1]] = [relation[0]]
#
#
#     # save relevant timelines
#     schema_time_relations = []
#     relevant_subevent_relations = []
#     for timeline in timelines:
#         children = [False for i in timeline]
#         for i, e1 in enumerate(timeline):
#             for j in range(i+1, len(timeline)):
#                 e2 = timeline[j]
#                 # child, parent - not add to time relation
#                 if (e1, e2) in subevents:
#                     relevant_subevent_relations.append((e1, e2))
#                     children[j] = True
#                 # child, parent - not add to time relation
#                 elif (e2, e1) in subevents:
#                     relevant_subevent_relations.append((e2, e1))
#                     children[i] = True
#                 # parent, parent - add to time relation
#                 elif not children[i] and not children[j]:
#                     schema_time_relations.append((e1, e2))
#                 # child, child - need to check if they have the same parent, if so, add time relation
#                 elif children[i] and children[j]:
#                     if set(child_par_dict[e1]) & set(child_par_dict[e2]):
#                         schema_time_relations.append((e1, e2))
#                 else:
#                     continue
#     relevant_subevent_relations = sorted(list(set(relevant_subevent_relations).union(subevents)))
#     schema_time_relations = sorted(list(set(schema_time_relations)))
#
#
#     # logical relations: currently without taking arguments into account
#     AND_rel = []
#     OR_rel = []
#     for rel1 in schema_time_relations:
#         for rel2 in schema_time_relations:
#             if rel1 == rel2:
#                 continue
#             else:
#                 if rel1[0] == rel2[0]:
#                     if not (rel2[1], rel1[1]) in OR_rel and not (rel1[1], rel2[1]) in schema_time_relations \
#                             and not (rel2[1], rel1[1]) in schema_time_relations:
#                         OR_rel.append((rel1[1], rel2[1]))
#                 if rel1[1] == rel2[1]:
#                     if not (rel2[0], rel1[0]) in OR_rel and not (rel1[0], rel2[0]) in schema_time_relations \
#                             and not (rel2[0], rel1[0]) in schema_time_relations:
#                         OR_rel.append((rel1[0], rel2[0]))
#                 if rel1[1] == rel2[0] and rel1[0] == rel2[1]:
#                     if (rel1[0], rel1[1]) in AND_rel or (rel1[1], rel1[0]) in AND_rel:
#                         continue
#                     else:
#                         AND_rel.append((rel1[0], rel1[1]))
#
#     # adjust temporal relations according to AND and OR relations - delete from temporal every relation that appears
#     # in logical and complete logical to make it transitive
#     # starting with making logical relation transitive
#     AND_REL = list(resolve_transitivity(AND_rel))
#     OR_REL = list(resolve_transitivity(OR_rel))
#     # if a relation in temporal appears in AND_REL, pop from temporal:
#     temporal_REL = []
#     for item in schema_time_relations:
#         if not item in AND_REL and not (item[1], item[0]) in AND_REL:
#             temporal_REL.append(item)
#     # if a couple appears in AND and in OR, then it should be only in AND:
#     for item in AND_REL:
#         if item in OR_REL:
#             OR_REL.pop(OR_REL.index(item))
#         if (item[1], item[0]) in OR_REL:
#             OR_REL.pop(OR_REL.index((item[1], item[0])))
#     # if a couple appears in OR and in temporal, then it should be only in temporal:
#     for item in temporal_REL:
#         if item in OR_REL:
#             OR_REL.pop(OR_REL.index(item))
#         if (item[1], item[0]) in OR_REL:
#             OR_REL.pop(OR_REL.index((item[1], item[0])))
#
#
#     subevent_REL = []
#     for item in relevant_subevent_relations:
#         if not item in AND_REL and not (item[1], item[0]) in AND_REL:
#             subevent_REL.append(item)
#
#     # print('schema temporal rel: ', sorted(temporal_REL))
#     # print('schema subevent rel: ', sorted(subevent_REL))
#     # print('schema OR rel: ', sorted(OR_REL))
#     # print('schema AND rel: ', sorted(AND_REL))
#
#     schema = {'temporal': temporal_REL, 'subevent': subevent_REL, 'OR': OR_REL, 'AND': AND_REL}
#
#     return schema


def construct_longest_chains(input):
    chains = []
    # constructing longest chains if seeing all relations up to length 4
    # looking for (A,B), (B,nothing) to construct A->B
    # looking for (A,B),(A,C),(B,C) triplets to construct a timeline A->B->C
    # looking for (A,B),(A,C),(A,D),(B,C),(B,D),(C,D) quadruplets to construct A->B->C->D

    keys = input.keys()
    checked_keys = []
    for A in keys:
        if A in checked_keys:
            continue
        for B in input[A]:
            if B in checked_keys:
                continue
            if B in keys:
                checked_keys.append(B)
                C = list(set(input[B]) & set(input[A]))
                if C:  # check to see if i have (A,C) for every C that we have (B,C)
                    for c in C:
                        if c in keys:
                            D = list(set(C) & set(input[c]))
                            if D:
                                for d in D:
                                    chains.append((A, B, c, d))
                        else:
                            chains.append((A, B, c))
                        checked_keys.append(c)
            else:
                chains.append((A, B))
            checked_keys.append(A)
    return chains


def parse_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith('top 30 events:'):
            events_start = i + 2
        if line.startswith('top 30 temporal relations:'):
            temporal_starts = i + 2
        if line.startswith('top 30 subevent relations:'):
            subevent_starts = i + 2
        if line.startswith('top 30 coref relations:'):
            coref_starts = i + 2
    events_ends = temporal_starts - 3
    temporal_ends = subevent_starts - 3
    subevent_ends = coref_starts - 3
    coref_ends = len(lines)

    document_temp_relation_dict = {}
    for i in range(temporal_starts, temporal_ends):
        rel_string = lines[i]
        pattern = "'\('(.*?)'\)'"
        substring = re.search(pattern, rel_string).group(1)
        relation = tuple(substring.split('\', \''))
        if 'occur' in relation:
            continue
        documents = rel_string.split(': {')[1][:-2]
        list_of_docs = documents.split(', ')
        if check_4_support_noun_verbs(file_name, relation, list_of_docs):
            continue
        for doc in list_of_docs:
            if int(doc) in document_temp_relation_dict.keys():
                document_temp_relation_dict[int(doc)].append(relation)
            else:
                document_temp_relation_dict[int(doc)] = [relation]

    document_subevent_relation_dict = {}
    for i in range(subevent_starts, subevent_ends):
        rel_string = lines[i]
        pattern = "'\('(.*?)'\)'"
        substring = re.search(pattern, rel_string).group(1)
        relation = tuple(substring.split('\', \''))
        documents = rel_string.split(': {')[1][:-2]
        list_of_docs = documents.split(', ')
        if check_4_support_noun_verbs(file_name, relation, list_of_docs):
            continue
        for doc in list_of_docs:
            if int(doc) in document_subevent_relation_dict.keys():
                document_subevent_relation_dict[int(doc)].append(relation)
            else:
                document_subevent_relation_dict[int(doc)] = [relation]

    document_coref_relation_dict = {}
    for i in range(coref_starts, coref_ends):
        rel_string = lines[i]
        pattern = "'\('(.*?)'\)'"
        substring = re.search(pattern, rel_string).group(1)
        relation = tuple(substring.split('\', \''))
        documents = rel_string.split(': {')[1][:-2]
        list_of_docs = documents.split(', ')
        if check_4_support_noun_verbs(file_name, relation, list_of_docs):
            continue
        for doc in list_of_docs:
            if int(doc) in document_coref_relation_dict.keys():
                document_coref_relation_dict[int(doc)].append(relation)
            else:
                document_coref_relation_dict[int(doc)] = [relation]

    document_verbs_and_arguments = []
    for i in range(events_start, events_ends):
        rel_string = lines[i]
        event = rel_string.split('\'')[1]
        arguments = rel_string.split('arguments: ')[1]
        document_verbs_and_arguments.append({event: arguments})

    return document_temp_relation_dict, document_subevent_relation_dict, document_coref_relation_dict, document_verbs_and_arguments


def add_preds_and_args(schema, document_verbs_and_arguments):
    schema['pred_and_args'] = []
    for item in document_verbs_and_arguments:
        verb = list(item.keys())[0]
        found_it = False
        for temp_rel in schema['temporal']:
            if verb in temp_rel:
                schema['pred_and_args'].append(item)
                found_it = True
                break
        if found_it:
            continue
        else:
            for subevent_rel in schema['subevent']:
                if verb in subevent_rel:
                    schema['pred_and_args'].append(item)
                    found_it = True
                    break

    return schema


def schema_generator(file_name):
    # file_name = 'output_direct/Business Change.txt'
    document_temp_relation_dict, document_subevent_relation_dict, document_coref_relation_dict, document_verbs_and_arguments = parse_file(file_name)

    timelines = {}
    for document in document_temp_relation_dict.keys():
        input4temporal = {}
        for item in document_temp_relation_dict[document]:
            if item[0] in input4temporal.keys():
                input4temporal[item[0]].append(item[1])
            else:
                input4temporal[item[0]] = [item[1]]

        timelines[document] = construct_longest_chains(input4temporal)

    # find intersecting timelines:
    freq_timelines = {}
    for doc in timelines.keys():
        for item in timelines[doc]:
            if item in freq_timelines:
                freq_timelines[item] += 1
            else:
                freq_timelines[item] = 1

    sorted_timelines = sorted(freq_timelines)

    # filter our list that contain other lists
    filtered_timelines = []
    for item in sorted_timelines:
        flag = True
        for item2 in sorted_timelines:
            if item == item2:
                continue
            if set(item).issubset(item2):  # doesn't take order into account
                # check order - only if order match then flag = False
                inds_in_items2 = []
                for i in item:
                    inds_in_items2.append(item2.index(i))
                if sorted(inds_in_items2) == inds_in_items2:
                    freq_timelines[item2] += 1
                    flag = False
                    break
        if flag:
            filtered_timelines.append(item)

    subevent_relations = build_hierarchy(document_subevent_relation_dict, filtered_timelines)

    coref_relations = []
    schema = build_schema(filtered_timelines, subevent_relations, coref_relations)

    schema_2 = fix_schema(schema)

    final_schema = add_preds_and_args(schema_2, document_verbs_and_arguments)
    # print(final_schema['temporal'])
    # print(final_schema['subevent'])
    return final_schema


if __name__ == '__main__':
    # schem = main('output/Kidnapping.txt')
    #
    # print('events: ', schem['pred_and_args'])
    # for event in schem['pred_and_args']:
    #     print(event)
    # print('temporal: ', schem['temporal'])
    # print('subevent: ', schem['subevent'])
    # print('OR: ', schem['OR'])
    # print('AND: ', schem['AND'])

    # directory = 'output_Typing_OnePass'
    file_names = ['output/Bombing Attacks.txt', 'output_direct/Bombing Attacks.txt',
                  'output/Business Change.txt', 'output_direct/Business Change.txt',
                  'output/Civil Unrest.txt', 'output_direct/Civil Unrest.txt',
                  'output/Disaster and Rescue.txt', 'output_direct/Disaster and Rescue.txt',
                  'output/Election.txt', 'output_direct/Election.txt',
                  'output/International Conflict.txt', 'output_direct/International Conflict.txt',
                  'output/Kidnapping.txt', 'output_direct/Kidnapping.txt',
                  'output/Mass Shooting.txt', 'output_direct/Mass Shooting.txt',
                  'output/Pandemic Outbreak.txt', 'output_direct/Pandemic Outbreak.txt',
                  'output/Sports Games.txt', 'output_direct/Sports Games.txt',
                  'output/Terrorism Attacks.txt', 'output_direct/Terrorism Attacks.txt']
    schemas = []
    for i, filename in enumerate(file_names):
        print(filename)
        schema = schema_generator(filename)
        if 'direct' in filename.split('/')[0]:
            schema['topic'] = filename.split('.')[0] + '_direct'
        else:
            schema['topic'] = filename.split('.')[0]
        schemas.append(schema)

    # print('finished')
    with open('KAIROS_schemas.txt', 'w') as outfile:
        json.dump(schemas, outfile)
