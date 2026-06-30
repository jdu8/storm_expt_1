## Supported datasets

Datasets should go into the ```data/``` folder.

### Youtube:

The original URL (http://dcomp.sor.ufscar.br/talmeida/youtubespamcollection) is not active anymore.
The original files are archived at (https://doi.org/10.24432/C58885).

For easy reproducibility, we provide our re-formatted files in ```data/youtube```.

### SMS

The original URL (http://www.dt.fee.unicamp.br/~tiago/smsspamcollection) is not active anymore.
The original files are archived at (https://doi.org/10.24432/C5CC84).

For easy reproducibility, we provide our re-formatted files in ```data/sms```.

### GLUE

The GLUE tasks and links to their files are listed on (https://gluebenchmark.com/tasks).

```
wget https://dl.fbaipublicfiles.com/glue/data/CoLA.zip
unzip CoLA.zip -d CoLA
wget https://dl.fbaipublicfiles.com/glue/data/RTE.zip
unzip RTE -d RTE
mkdir MRPC
python download_mrpc.py
```

### MultiWOZ 2.1

```
git clone https://github.com/budzianowski/multiwoz.git
unzip multiwoz/data/MultiWOZ_2.1.zip -d multiwoz/data/
python split_multiwoz_data.py --data_dir multiwoz/data/MultiWOZ_2.1
```

### MultiWOZ 2.4

```
git clone https://github.com/smartyfh/MultiWOZ2.4.git
unzip MultiWOZ2.4/data/MULTIWOZ2.4.zip -d MultiWOZ2.4/data/
python split_multiwoz_data.py --data_dir MultiWOZ2.4/data/MULTIWOZ2.4
```

## Add symmetric noise

```
python corrupt_data.py --file_in youtube/train.tsv --file_out youtube/train_corrupt_30.tsv --task youtube --corruption 0.3 --seed 42
python corrupt_data.py --file_in sms/train.tsv --file_out sms/train_corrupt_30.tsv --task sms --corruption 0.3 --seed 42
python corrupt_data.py --file_in CoLA/train.tsv --file_out CoLA/train_corrupt_30.tsv --task cola --corruption 0.3 --seed 42
python corrupt_data.py --file_in MRPC/train.tsv --file_out MRPC/train_corrupt_30.tsv --task mrpc --corruption 0.3 --seed 42
python corrupt_data.py --file_in RTE/train.tsv --file_out RTE/train_corrupt_30.tsv --task rte --corruption 0.3 --seed 42
```

## TF-IDF features

The TF-IDF features for the Youtube and SMS datasets used for our experiments were
computed with the help of the AGRA code base (https://github.com/anasedova/AGRA).

For easy reproducibility, we provide our computed TF-IDF features as pickle files, found in the
respective data folders.

```
gunzip youtube/tfidf_feats/*.gz
gunzip sms/tfidf_feats/*.gz
```
