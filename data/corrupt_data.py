# coding=utf-8

import argparse
import json
import numpy as np
from transformers import (set_seed)


DATA_SPECS = {'youtube': {'has_header': True, 'label': 2, 'sentence1': 1, 'sentence2': None, 'num_labels': 2, 'label_map': None},
              'sms': {'has_header': True, 'label': 2, 'sentence1': 1, 'sentence2': None, 'num_labels': 2, 'label_map': None},
              'cola': {'has_header': False, 'label': 1, 'sentence1': 3, 'sentence2': None, 'num_labels': 2, 'label_map': None},
              'mrpc': {'has_header': True, 'label': 0, 'sentence1': 3, 'sentence2': 4, 'num_labels': 2, 'label_map': None},
              'rte': {'has_header': True, 'label': 3, 'sentence1': 1, 'sentence2': 2, 'num_labels': 2, 'label_map': {'not_entailment': 0, 'entailment': 1, 0: 'not_entailment', 1: 'entailment'}},
}


def load_raw_dataset(input_file, data_specs):
    raw_data = []
    with open(input_file, "r", encoding='utf-8') as f:
        header = None
        for l_itr, line in enumerate(f):
            if data_specs['has_header'] and l_itr == 0:
                header = line.strip()
                continue
            raw_data_point = line.strip().split('\t')
            raw_data.append(raw_data_point)
    return raw_data, header


def corrupt_data(dataset, corruption_rate, data_specs):
    def label_map(label, lbl_map):
        if lbl_map is not None:
            return lbl_map[label]
        else:
            return label

    is_corrupted = []
    corruption_prob = np.random.random_sample(len(dataset))
    for i_itr, i in enumerate(dataset):
        label = int(label_map(dataset[i_itr][data_specs['label']], data_specs['label_map']))
        if corruption_prob[i_itr] <= corruption_rate:
            if data_specs['num_labels'] == 1:
                raise NotImplementedError()
            elif data_specs['num_labels'] == 2:
                rn = int(not label) # Flips a binary value
            else:
                rn = np.random.choice([e for e in range(data_specs['num_labels']) if e != label])
            dataset[i_itr][data_specs['label']] = str(label_map(rn, data_specs['label_map']))
            is_corrupted.append(True)
        else:
            is_corrupted.append(False)
    return is_corrupted


def main():
    parser = argparse.ArgumentParser(description="...")
    
    # Required parameters
    parser.add_argument("--file_in", type=str, default=None, required=True, help="")
    parser.add_argument("--file_out", type=str, default=None, required=True, help="")
    parser.add_argument("--task", type=str, default=None, required=True, help="")
    parser.add_argument("--corruption", type=float, default=0.0, required=True, help="")
    parser.add_argument("--seed", type=int, default=42, help="")

    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    raw_data, header = load_raw_dataset(args.file_in, DATA_SPECS[args.task])

    is_corrupted = corrupt_data(raw_data, args.corruption, data_specs=DATA_SPECS[args.task])

    if header is not None:
        header += '\tCorrupted'    
    with open(args.file_out, "w") as f:
        if header is not None:
            f.write(header + '\n')
        for i_itr, i in enumerate(raw_data):
            f.write('\t'.join(i + [str(is_corrupted[i_itr])]) + '\n')


if __name__ == "__main__":
    main()
