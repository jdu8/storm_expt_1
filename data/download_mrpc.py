# coding=utf-8
#
# Adopted from https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py
# (not accessible anymore as of 2024)

import argparse
import os
import urllib.request

MRPC_TRAIN = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt'
MRPC_DEV = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc'
MRPC_TEST = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt'


def format_mrpc(data_dir):
    mrpc_train_file = os.path.join(data_dir, "msr_paraphrase_train.txt")
    mrpc_test_file = os.path.join(data_dir, "msr_paraphrase_test.txt")
    urllib.request.urlretrieve(MRPC_TRAIN, mrpc_train_file)
    urllib.request.urlretrieve(MRPC_TEST, mrpc_test_file)
    urllib.request.urlretrieve(MRPC_DEV, os.path.join(data_dir, "dev_ids.tsv"))

    dev_ids = []
    with open(os.path.join(data_dir, "dev_ids.tsv"), encoding="utf8") as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split('\t'))

    with open(mrpc_train_file, encoding="utf8") as data_fh, \
         open(os.path.join(data_dir, "train.tsv"), 'w', encoding="utf8") as train_fh, \
         open(os.path.join(data_dir, "dev.tsv"), 'w', encoding="utf8") as dev_fh:
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split('\t')
            if [id1, id2] in dev_ids:
                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            else:
                train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

    with open(mrpc_test_file, encoding="utf8") as data_fh, \
            open(os.path.join(data_dir, "test.tsv"), 'w', encoding="utf8") as test_fh:
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split('\t')
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))


def main():
    parser = argparse.ArgumentParser("Download and prepare MRPC.")
    parser.add_argument('--data_dir', type=str, default="MRPC",
                        help='directory to save data to')
    args = parser.parse_args()

    format_mrpc(args.data_dir)


if __name__ == "__main__":
    main()
