# Unifying Token and Span Level Supervisions for Few-Shot Sequence Labeling
Code of Our Paper "Unifying Token and Span Level Supervisions for Few-Shot Sequence Labeling"


## Enviroment
```
pip install -r requirements.txt
```


## Dataset
```
mkdir data
cd data
```
### FewNERD
We use the latest sampled dataset from [FewNERD](https://cloud.tsinghua.edu.cn/f/0e38bd108d7b49808cc4/?dl=1), which corresponds to the results of [the 6-th version FewNERD paper](https://arxiv.org/pdf/2105.07464v6.pdf).
The new sampled data fixs the data sampling bug, see [issue](https://github.com/thunlp/Few-NERD/issues/15).

```
wget -O data.zip https://cloud.tsinghua.edu.cn/f/0e38bd108d7b49808cc4/?dl=1
unzip data.zip
mv episode-data/* ./
rm -rf episide-data
```

### SNIPS and Cross-Dataset
We use the sampled data from [SNIPS-FewShot](https://atmahou.github.io/attachments/ACL2020data.zip).

```
wget https://atmahou.github.io/attachments/ACL2020data.zip
unzip ACL2020data.zip
mv ACL2020data/* ./
rm -rf ACL2020data
cd ..
```


## Train and Evaluation

Make sure your have the data file structure as follows:
```
├── bash
│   ├── fewnerd
│   └── snips
├── checkpoint
├── data
│   ├── inter
│   ├── intra
│   ├── xval_snips
│   ├── xval_snips_shot_5
│   ├── xval_ner
│   └── xval_ner_shot_5
├── model
│   ├── CDAP.py
│   └── utils.py
├── README.md
├── requirements.txt
├── results
├── train_demo.py
└── util
    ├── data_loader.py
    ├── framework.py
    └── utils.py
```


### FewNERD
```
bash bash/fewnerd/run_mode.sh [gpu_id] [mode] [N] [K]
    - mode: intra/inter
    - N, K: 5 1, 5 5, 10 1
    e.g., bash bash/fewnerd/run_mode.sh 0 inter 5 1
bash bash/fewnerd/10wat_5shot_mode.sh [gpu_id] [mode]
    - mode: intra/inter
    e.g., bash/fewnerd/10wat_5shot_mode.sh 0 inter
```
### SNIPS
```
bash bash/snips/1-shot/1_shot_mode_1.sh [gpu_id]
...
bash bash/snips/1-shot/1_shot_mode_7.sh [gpu_id]
```
```
bash bash/snips/5-shot/5_shot_mode_1.sh [gpu_id]
...
bash bash/snips/5-shot/5_shot_mode_7.sh [gpu_id]
```

### Cross
```
bash bash/ner/1-shot/1_shot_mode_1.sh [gpu_id]
...
bash bash/ner/1-shot/1_shot_mode_4.sh [gpu_id]
```
```
bash bash/ner/5-shot/5_shot_mode_1.sh [gpu_id]
...
bash bash/ner/5-shot/5_shot_mode_4.sh [gpu_id]
```


# Acknowledgments
We thank all authors from this paper: '[An Enhanced Span-based Decomposition Method for Few-Shot Sequence Labeling](https://github.com/Wangpeiyi9979/ESD)'. We adopt many codes from their projects.

# Contact
If there are any questions, please feel free to contact me: Zifeng Cheng (chengzf@smail.nju.edu.cn).
