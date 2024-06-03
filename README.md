# Learning Object State Changes in Videos: An Open-World Perspective 
[**Learning Object State Changes in Videos: An Open-World Perspective**](https://arxiv.org/abs/2312.11782)                                     
Zihui Xue, Kumar Ashutosh, Kristen Grauman  
CVPR, 2024  
[project page](https://vision.cs.utexas.edu/projects/VidOSC/) | [arxiv](https://arxiv.org/abs/2312.11782) | [bibtex](#citation)

## HowToChange dataset
**HowToChange (Evaluation)**: we collect temporal annotations for 5,423 video clips sourced from HowTo100M, encompassing 409 OSCs (20 state transitions associated with 134 objects). 
<p align="left">
  <img src="images/howtochange_annotation.png" width=60%>
</p>

**HowToChange (Pseudo-labeled Train)**: we propose an automated data collection process (paper Section 3.3) and identify 36,075 video clips that may contain OSC, using ASR transcriptions and LLAMA2.
<p align="left">
  <img src="images/howtochange_miningforosc.png" width=50%>
</p>

### Data files
+ See [data_files/osc_split.csv](data_files/osc_split.csv) for the complete list of OSCs and seen/novel splits.  
+ See [data_files/howtochange_eval.csv](data_files/howtochange_eval.csv) for:
  + video clip information (first 3 columns, YouTube id, start time and duration)
  + temporal annotations (middle 3 columns, time ranges for the initial, transistioning and end state of OSC)
  + OSC associated with the clip (last 2 columns: OSC name and whether this is novel OSC)
+ See [data_files/howtochange_unlabeled_train.csv](data_files/howtochange_unlabeled_train.csv) for:
  + video clip information (first 3 columns, YouTube id, start time and duration)
  + OSC associated with the clip (last 2 columns: OSC name and whether this is novel OSC, last column always False since all train OSCs are seen)

### Utils
+ Run [data_scripts/preprocess_clip.py](data_scripts/preprocess_clip.py) to extract these clips. Make sure to have [yt-dlp](https://github.com/yt-dlp/yt-dlp) and [ffmpeg](https://ffmpeg.org) installed.  
+ Run [data_scripts/pseudo_label.py](data_scripts/pseudo_label.py) to pseudo label training clips with CLIP/VideoCLIP. To use VideoCLIP, make sure to have the [MMPT toolkit](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT) installed.
+ See [data_scripts/read_ann.py](data_scripts/read_ann.py) for helper functions to extract frames and read the annotations.


### Evaluation
+ Run [data_scripts/evaluator.py](data_scripts/evaluator.py), remember to replace `predict()` with your model's prediction function. The evaluator is set as 1fps.

## VidOSC code

### Before you start
1. set up the environment
```bash
conda env create --file environment.yml
conda activate vidosc
```

2. update data path in [train.py](train.py):   
`--feat_dir`: path to InternVideo features, we use the precomputed InternVideo-MM-L14 features [here](https://github.com/TengdaHan/TemporalAlignNet/tree/main/htm_zoo)   
`--pseudolabel_dir`: path to pseudo labels given by clip / videoclip (run [data_scripts/pseudo_label.py](data_scripts/pseudo_label.py) to generate pseudo labels) 

Data Structure:
```
feat_dir
  |- VIDEO_ID.pth.tar
pseudolabel_dir
  |- clip_probs
  |- videoclip_probs
```

### Training
```bash
# Train one model for one state transition
sc_list=("chopping" "slicing" "frying" "peeling" "blending" "roasting" "browning" "grating" "grilling" "crushing" "melting" "squeezing" "sauteing" "shredding" "whipping" "rolling" "mashing" "mincing" "coating" "zesting") 
for sc in ${sc_list[@]}; do
  python train.py --use_videoclip --det 1 --sc_list $sc
done

# Train one multitask model for all state transitions (see paper Table 7)
python train.py --use_videoclip --det 1 --sc_list all
```

`--use_videoclip`: use VideoCLIP as the pseudo labeler.  
`--det`: 0: InternVideo features, 1: global InternVideo + object-centric Internvideo features as input (paper Section 3.2)    
`--sc_list`: specify the state transition to train, using `all` trains a multitask model for all state transitions.

### Evaluation
```bash
python train.py --use_videoclip --det 1 --sc_list $SC_NAME --ckpt $CKPT_PATH
```

`--ckpt`: path to the model checkpoint.   
`--sc_list`: specify the state transition to evaluate.


## License
VidOSC is licensed under the [CC-BY-NC license](LICENSE).

## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@article{xue2023learning,
  title={Learning Object State Changes in Videos: An Open-World Perspective},
  author={Xue, Zihui and Ashutosh, Kumar and Grauman, Kristen},
  journal={CVPR},
  year={2024}
}
``` 
