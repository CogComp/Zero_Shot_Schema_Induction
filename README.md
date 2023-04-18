# Zero-Shot On-the-Fly Event Schema Induction

This is the repository for the resources in EACL 2023 (findings) Paper "[Zero-Shot On-the-Fly Event Schema Induction](https://arxiv.org/pdf/2210.06254.pdf)". This repository contains the source code and datasets used in our paper.

## Abstract

What are the events involved in a pandemic outbreak? What steps should be taken when planning a wedding? The answers to these questions can be found by collecting many documents on the complex event of interest, extracting relevant information, and analyzing it. We present a new approach in which large language models are utilized to generate source documents that allow predicting, given a high-level event definition, the specific events, arguments, and relations between them to construct a schema that describes the complex event in its entirety. Using our model, complete schemas on any topic can be generated on-the-fly without any manual data collection, i.e., in a zero-shot manner. Moreover, we develop efficient methods to extract pertinent information from texts and demonstrate in a series of experiments that these schemas are considered to be more complete than human-curated ones in the majority of examined scenarios. Finally, we show that this framework is comparable in performance with previous supervised schema induction methods that rely on collecting real texts while being more general and flexible without the need for a predefined ontology.

<p align="center">
    <img src="https://github.com/CogComp/Zero_Shot_Schema_Induction/blob/main/example.png?raw=true" alt="drawing" width="500"/>
</p>

## Ports used for IE services

6003: constituency parsing - a model developed by allennlp<br>
6004: subevent - OnePass model trained on HiEve<br>
6009: temporal - OnePass model trained on MATRES<br>
8888: coref (not provided)<br>

## Environment

cp_env.yml specifies the conda environment used to run constituency parsing service<br>
main_env.yml specifies the conda environment used to run the temporal and subevent services, and also the main experiment<br>

## How to run the code

```conda activate cp_env```<br>
```nohup python3 constituency_parsing.py --port 6003 > cp.out 2>&1 &```<br>
```conda deactivate```<br>
```conda activate main_env```<br>
```cd LEC_OnePass/```<br>
```mkdir model_params```<br>
```cd model_params```<br>
```mkdir MATRES_best```<br>
```mkdir HiEve_best```<br>
```cp /path/to/MATRES_MODEL MATRES_best/```<br>
```cp /path/to/HIEVE_MODEL HiEve_best/```<br>
```cd ../```<br>
```nohup python backend.py --port 6004 > backend.out 2>&1 &```<br>
```nohup python backend_t.py --port 6009 > backend_t.out 2>&1 &```<br>
```cd ../```<br>
```nohup python Information_Extractor.py > ie.out 2>&1 &```<br>

## Curl example for API calls

Call temporal api (curl example, on dickens 4024, not suggested due to speed issue, openweb):
```curl --request POST --data '{"corpusId": "", "id": "", "text": "In the January attack, two Palestinian suicide bombers blew themselves up in central Tel Aviv. The bombing destroyed the whole building, killing 23 other people.", "tokens": ["In", "the", "January", "attack", ",", "two", "Palestinian", "suicide", "bombers", "blew", "themselves", "up", "in", "central", "Tel", "Aviv", ".", "The", "bombing", "destroyed", "the", "whole", "building", ",", "killing", "23", "other", "people", "."], "sentences": {"generator": "srl_pipeline", "score": 1.0, "sentenceEndPositions": [17, 29]}, "views": [{"viewName": "TOKENS", "viewData": {"viewType": "edu.illinois.cs.cogcomp.core.datastructures.textannotation.TokenLabelView", "viewName": "TOKENS", "generator": "Cogcomp-SRL", "score": 1.0, "constituents": [{"label": "In", "score": 1.0, "start": 0, "end": 1}, {"label": "the", "score": 1.0, "start": 1, "end": 2}, {"label": "January", "score": 1.0, "start": 2, "end": 3}, {"label": "attack", "score": 1.0, "start": 3, "end": 4}, {"label": ",", "score": 1.0, "start": 4, "end": 5}, {"label": "two", "score": 1.0, "start": 5, "end": 6}, {"label": "Palestinian", "score": 1.0, "start": 6, "end": 7}, {"label": "suicide", "score": 1.0, "start": 7, "end": 8}, {"label": "bombers", "score": 1.0, "start": 8, "end": 9}, {"label": "blew", "score": 1.0, "start": 9, "end": 10}, {"label": "themselves", "score": 1.0, "start": 10, "end": 11}, {"label": "up", "score": 1.0, "start": 11, "end": 12}, {"label": "in", "score": 1.0, "start": 12, "end": 13}, {"label": "central", "score": 1.0, "start": 13, "end": 14}, {"label": "Tel", "score": 1.0, "start": 14, "end": 15}, {"label": "Aviv", "score": 1.0, "start": 15, "end": 16}, {"label": ".", "score": 1.0, "start": 16, "end": 17}, {"label": "The", "score": 1.0, "start": 17, "end": 18}, {"label": "bombing", "score": 1.0, "start": 18, "end": 19}, {"label": "destroyed", "score": 1.0, "start": 19, "end": 20}, {"label": "the", "score": 1.0, "start": 20, "end": 21}, {"label": "whole", "score": 1.0, "start": 21, "end": 22}, {"label": "building", "score": 1.0, "start": 22, "end": 23}, {"label": ",", "score": 1.0, "start": 23, "end": 24}, {"label": "killing", "score": 1.0, "start": 24, "end": 25}, {"label": "23", "score": 1.0, "start": 25, "end": 26}, {"label": "other", "score": 1.0, "start": 26, "end": 27}, {"label": "people", "score": 1.0, "start": 27, "end": 28}, {"label": ".", "score": 1.0, "start": 28, "end": 29}]}}, {"viewName": "Event_extraction", "viewData": [{"viewType": "edu.illinois.cs.cogcomp.core.datastructures.textannotation.PredicateArgumentView", "viewName": "event_extraction", "generator": "cogcomp_kairos_event_ie_v1.0", "score": 1.0, "constituents": [{"label": "Conflict:Attack:Unspecified", "score": 1.0, "start": 3, "end": 4, "properties": {"SenseNumber": "01", "sentence_id": 0, "predicate": ["attack"]}}, {"label": "Attacker", "score": 1.0, "start": 6, "end": 7, "entity_type": "gpe"}, {"label": "Conflict:Attack:Unspecified", "score": 1.0, "start": 8, "end": 9, "properties": {"SenseNumber": "01", "sentence_id": 0, "predicate": ["bombers"]}}, {"label": "Attacker", "score": 1.0, "start": 8, "end": 9, "entity_type": "per"}, {"label": "Target", "score": 1.0, "start": 6, "end": 7, "entity_type": "gpe"}, {"label": "Disaster:FireExplosion:Unspecified", "score": 1.0, "start": 18, "end": 19, "properties": {"SenseNumber": "01", "sentence_id": 1, "predicate": []}}, {"label": "Place", "score": 1.0, "start": 22, "end": 23, "entity_type": "fac"}], "relations": [{"relationName": "Attacker", "srcConstituent": 0, "targetConstituent": 1}, {"relationName": "Attacker", "srcConstituent": 2, "targetConstituent": 3}, {"relationName": "Target", "srcConstituent": 2, "targetConstituent": 4}, {"relationName": "Place", "srcConstituent": 5, "targetConstituent": 6}]}]}]}' -H "Content-type: application/json" http://dickens.seas.upenn.edu:4024/annotate```

Call subevent api (curl example, on holst 6004):
```curl --request POST --data '{"corpusId": "", "id": "", "text": "In the January attack, two Palestinian suicide bombers blew themselves up in central Tel Aviv. The bombing destroyed the whole building, killing 23 other people.", "tokens": ["In", "the", "January", "attack", ",", "two", "Palestinian", "suicide", "bombers", "blew", "themselves", "up", "in", "central", "Tel", "Aviv", ".", "The", "bombing", "destroyed", "the", "whole", "building", ",", "killing", "23", "other", "people", "."], "sentences": {"generator": "srl_pipeline", "score": 1.0, "sentenceEndPositions": [17, 29]}, "views": [{"viewName": "TOKENS", "viewData": {"viewType": "edu.illinois.cs.cogcomp.core.datastructures.textannotation.TokenLabelView", "viewName": "TOKENS", "generator": "Cogcomp-SRL", "score": 1.0, "constituents": [{"label": "In", "score": 1.0, "start": 0, "end": 1}, {"label": "the", "score": 1.0, "start": 1, "end": 2}, {"label": "January", "score": 1.0, "start": 2, "end": 3}, {"label": "attack", "score": 1.0, "start": 3, "end": 4}, {"label": ",", "score": 1.0, "start": 4, "end": 5}, {"label": "two", "score": 1.0, "start": 5, "end": 6}, {"label": "Palestinian", "score": 1.0, "start": 6, "end": 7}, {"label": "suicide", "score": 1.0, "start": 7, "end": 8}, {"label": "bombers", "score": 1.0, "start": 8, "end": 9}, {"label": "blew", "score": 1.0, "start": 9, "end": 10}, {"label": "themselves", "score": 1.0, "start": 10, "end": 11}, {"label": "up", "score": 1.0, "start": 11, "end": 12}, {"label": "in", "score": 1.0, "start": 12, "end": 13}, {"label": "central", "score": 1.0, "start": 13, "end": 14}, {"label": "Tel", "score": 1.0, "start": 14, "end": 15}, {"label": "Aviv", "score": 1.0, "start": 15, "end": 16}, {"label": ".", "score": 1.0, "start": 16, "end": 17}, {"label": "The", "score": 1.0, "start": 17, "end": 18}, {"label": "bombing", "score": 1.0, "start": 18, "end": 19}, {"label": "destroyed", "score": 1.0, "start": 19, "end": 20}, {"label": "the", "score": 1.0, "start": 20, "end": 21}, {"label": "whole", "score": 1.0, "start": 21, "end": 22}, {"label": "building", "score": 1.0, "start": 22, "end": 23}, {"label": ",", "score": 1.0, "start": 23, "end": 24}, {"label": "killing", "score": 1.0, "start": 24, "end": 25}, {"label": "23", "score": 1.0, "start": 25, "end": 26}, {"label": "other", "score": 1.0, "start": 26, "end": 27}, {"label": "people", "score": 1.0, "start": 27, "end": 28}, {"label": ".", "score": 1.0, "start": 28, "end": 29}]}}, {"viewName": "Event_extraction", "viewData": [{"viewType": "edu.illinois.cs.cogcomp.core.datastructures.textannotation.PredicateArgumentView", "viewName": "event_extraction", "generator": "cogcomp_kairos_event_ie_v1.0", "score": 1.0, "constituents": [{"label": "Conflict:Attack:Unspecified", "score": 1.0, "start": 3, "end": 4, "properties": {"SenseNumber": "01", "sentence_id": 0, "predicate": ["attack"]}}, {"label": "Attacker", "score": 1.0, "start": 6, "end": 7, "entity_type": "gpe"}, {"label": "Conflict:Attack:Unspecified", "score": 1.0, "start": 8, "end": 9, "properties": {"SenseNumber": "01", "sentence_id": 0, "predicate": ["bombers"]}}, {"label": "Attacker", "score": 1.0, "start": 8, "end": 9, "entity_type": "per"}, {"label": "Target", "score": 1.0, "start": 6, "end": 7, "entity_type": "gpe"}, {"label": "Disaster:FireExplosion:Unspecified", "score": 1.0, "start": 18, "end": 19, "properties": {"SenseNumber": "01", "sentence_id": 1, "predicate": []}}, {"label": "Place", "score": 1.0, "start": 22, "end": 23, "entity_type": "fac"}], "relations": [{"relationName": "Attacker", "srcConstituent": 0, "targetConstituent": 1}, {"relationName": "Attacker", "srcConstituent": 2, "targetConstituent": 3}, {"relationName": "Target", "srcConstituent": 2, "targetConstituent": 4}, {"relationName": "Place", "srcConstituent": 5, "targetConstituent": 6}]}]}]}' -H "Content-type: application/json" http://localhost:6004/annotate```

Call temporal api (curl example, on holst 6009):
```curl --request POST --data '{"corpusId": "", "id": "", "text": "In the January attack, two Palestinian suicide bombers blew themselves up in central Tel Aviv. The bombing destroyed the whole building, killing 23 other people.", "tokens": ["In", "the", "January", "attack", ",", "two", "Palestinian", "suicide", "bombers", "blew", "themselves", "up", "in", "central", "Tel", "Aviv", ".", "The", "bombing", "destroyed", "the", "whole", "building", ",", "killing", "23", "other", "people", "."], "sentences": {"generator": "srl_pipeline", "score": 1.0, "sentenceEndPositions": [17, 29]}, "views": [{"viewName": "TOKENS", "viewData": {"viewType": "edu.illinois.cs.cogcomp.core.datastructures.textannotation.TokenLabelView", "viewName": "TOKENS", "generator": "Cogcomp-SRL", "score": 1.0, "constituents": [{"label": "In", "score": 1.0, "start": 0, "end": 1}, {"label": "the", "score": 1.0, "start": 1, "end": 2}, {"label": "January", "score": 1.0, "start": 2, "end": 3}, {"label": "attack", "score": 1.0, "start": 3, "end": 4}, {"label": ",", "score": 1.0, "start": 4, "end": 5}, {"label": "two", "score": 1.0, "start": 5, "end": 6}, {"label": "Palestinian", "score": 1.0, "start": 6, "end": 7}, {"label": "suicide", "score": 1.0, "start": 7, "end": 8}, {"label": "bombers", "score": 1.0, "start": 8, "end": 9}, {"label": "blew", "score": 1.0, "start": 9, "end": 10}, {"label": "themselves", "score": 1.0, "start": 10, "end": 11}, {"label": "up", "score": 1.0, "start": 11, "end": 12}, {"label": "in", "score": 1.0, "start": 12, "end": 13}, {"label": "central", "score": 1.0, "start": 13, "end": 14}, {"label": "Tel", "score": 1.0, "start": 14, "end": 15}, {"label": "Aviv", "score": 1.0, "start": 15, "end": 16}, {"label": ".", "score": 1.0, "start": 16, "end": 17}, {"label": "The", "score": 1.0, "start": 17, "end": 18}, {"label": "bombing", "score": 1.0, "start": 18, "end": 19}, {"label": "destroyed", "score": 1.0, "start": 19, "end": 20}, {"label": "the", "score": 1.0, "start": 20, "end": 21}, {"label": "whole", "score": 1.0, "start": 21, "end": 22}, {"label": "building", "score": 1.0, "start": 22, "end": 23}, {"label": ",", "score": 1.0, "start": 23, "end": 24}, {"label": "killing", "score": 1.0, "start": 24, "end": 25}, {"label": "23", "score": 1.0, "start": 25, "end": 26}, {"label": "other", "score": 1.0, "start": 26, "end": 27}, {"label": "people", "score": 1.0, "start": 27, "end": 28}, {"label": ".", "score": 1.0, "start": 28, "end": 29}]}}, {"viewName": "Event_extraction", "viewData": [{"viewType": "edu.illinois.cs.cogcomp.core.datastructures.textannotation.PredicateArgumentView", "viewName": "event_extraction", "generator": "cogcomp_kairos_event_ie_v1.0", "score": 1.0, "constituents": [{"label": "Conflict:Attack:Unspecified", "score": 1.0, "start": 3, "end": 4, "properties": {"SenseNumber": "01", "sentence_id": 0, "predicate": ["attack"]}}, {"label": "Attacker", "score": 1.0, "start": 6, "end": 7, "entity_type": "gpe"}, {"label": "Conflict:Attack:Unspecified", "score": 1.0, "start": 8, "end": 9, "properties": {"SenseNumber": "01", "sentence_id": 0, "predicate": ["bombers"]}}, {"label": "Attacker", "score": 1.0, "start": 8, "end": 9, "entity_type": "per"}, {"label": "Target", "score": 1.0, "start": 6, "end": 7, "entity_type": "gpe"}, {"label": "Disaster:FireExplosion:Unspecified", "score": 1.0, "start": 18, "end": 19, "properties": {"SenseNumber": "01", "sentence_id": 1, "predicate": []}}, {"label": "Place", "score": 1.0, "start": 22, "end": 23, "entity_type": "fac"}], "relations": [{"relationName": "Attacker", "srcConstituent": 0, "targetConstituent": 1}, {"relationName": "Attacker", "srcConstituent": 2, "targetConstituent": 3}, {"relationName": "Target", "srcConstituent": 2, "targetConstituent": 4}, {"relationName": "Place", "srcConstituent": 5, "targetConstituent": 6}]}]}]}' -H "Content-type: application/json" http://localhost:6009/annotate```

Call coref api (curl example, on holst 8888):
```curl -d '{"corpusId": "", "id": "", "text": "In the January attack, two Palestinian suicide bombers blew themselves up in central Tel Aviv. The bombing destroyed the whole building, killing 23 other people.", "tokens": ["In", "the", "January", "attack", ",", "two", "Palestinian", "suicide", "bombers", "blew", "themselves", "up", "in", "central", "Tel", "Aviv", ".", "The", "bombing", "destroyed", "the", "whole", "building", ",", "killing", "23", "other", "people", "."], "sentences": {"generator": "srl_pipeline", "score": 1.0, "sentenceEndPositions": [17, 29]}, "views": [{"viewName": "TOKENS", "viewData": {"viewType": "edu.illinois.cs.cogcomp.core.datastructures.textannotation.TokenLabelView", "viewName": "TOKENS", "generator": "Cogcomp-SRL", "score": 1.0, "constituents": [{"label": "In", "score": 1.0, "start": 0, "end": 1}, {"label": "the", "score": 1.0, "start": 1, "end": 2}, {"label": "January", "score": 1.0, "start": 2, "end": 3}, {"label": "attack", "score": 1.0, "start": 3, "end": 4}, {"label": ",", "score": 1.0, "start": 4, "end": 5}, {"label": "two", "score": 1.0, "start": 5, "end": 6}, {"label": "Palestinian", "score": 1.0, "start": 6, "end": 7}, {"label": "suicide", "score": 1.0, "start": 7, "end": 8}, {"label": "bombers", "score": 1.0, "start": 8, "end": 9}, {"label": "blew", "score": 1.0, "start": 9, "end": 10}, {"label": "themselves", "score": 1.0, "start": 10, "end": 11}, {"label": "up", "score": 1.0, "start": 11, "end": 12}, {"label": "in", "score": 1.0, "start": 12, "end": 13}, {"label": "central", "score": 1.0, "start": 13, "end": 14}, {"label": "Tel", "score": 1.0, "start": 14, "end": 15}, {"label": "Aviv", "score": 1.0, "start": 15, "end": 16}, {"label": ".", "score": 1.0, "start": 16, "end": 17}, {"label": "The", "score": 1.0, "start": 17, "end": 18}, {"label": "bombing", "score": 1.0, "start": 18, "end": 19}, {"label": "destroyed", "score": 1.0, "start": 19, "end": 20}, {"label": "the", "score": 1.0, "start": 20, "end": 21}, {"label": "whole", "score": 1.0, "start": 21, "end": 22}, {"label": "building", "score": 1.0, "start": 22, "end": 23}, {"label": ",", "score": 1.0, "start": 23, "end": 24}, {"label": "killing", "score": 1.0, "start": 24, "end": 25}, {"label": "23", "score": 1.0, "start": 25, "end": 26}, {"label": "other", "score": 1.0, "start": 26, "end": 27}, {"label": "people", "score": 1.0, "start": 27, "end": 28}, {"label": ".", "score": 1.0, "start": 28, "end": 29}]}}, {"viewName": "Event_extraction", "viewData": [{"viewType": "edu.illinois.cs.cogcomp.core.datastructures.textannotation.PredicateArgumentView", "viewName": "event_extraction", "generator": "cogcomp_kairos_event_ie_v1.0", "score": 1.0, "constituents": [{"label": "Conflict:Attack:Unspecified", "score": 1.0, "start": 3, "end": 4, "properties": {"SenseNumber": "01", "sentence_id": 0, "predicate": ["attack"]}}, {"label": "Attacker", "score": 1.0, "start": 6, "end": 7, "entity_type": "gpe"}, {"label": "Conflict:Attack:Unspecified", "score": 1.0, "start": 8, "end": 9, "properties": {"SenseNumber": "01", "sentence_id": 0, "predicate": ["bombers"]}}, {"label": "Attacker", "score": 1.0, "start": 8, "end": 9, "entity_type": "per"}, {"label": "Target", "score": 1.0, "start": 6, "end": 7, "entity_type": "gpe"}, {"label": "Disaster:FireExplosion:Unspecified", "score": 1.0, "start": 18, "end": 19, "properties": {"SenseNumber": "01", "sentence_id": 1, "predicate": []}}, {"label": "Place", "score": 1.0, "start": 22, "end": 23, "entity_type": "fac"}], "relations": [{"relationName": "Attacker", "srcConstituent": 0, "targetConstituent": 1}, {"relationName": "Attacker", "srcConstituent": 2, "targetConstituent": 3}, {"relationName": "Target", "srcConstituent": 2, "targetConstituent": 4}, {"relationName": "Place", "srcConstituent": 5, "targetConstituent": 6}]}]}]}' -H "Content-Type: application/json" -X POST http://localhost:8888/annotate```