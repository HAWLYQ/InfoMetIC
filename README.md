# InfoMetIC: An Informative Metric for Reference-free Image Caption Evaluation (ACL 2023)
By Anwen Hu, Shizhe Chen, Liang Zhang, Qin Jin


## Requirements
torch >= 1.7

## Evaluate Image Captioning results on MSCOCO
1.download preprocessed image features, put in ./infometic/data 

2.download model, put in ./infometic/save

3.run inference code as follows:

```
cd infometic
python inference.py --image {mscoco_image_name} --caption {caption}
```
for example,
```
python inference.py --image 'COCO_val2014_000000197461.jpg' --caption ' A very large sheep is standing under clouds.'
```

## Training and Evaluation on benchmarks are on the way...


## Citation
if you find this code useful for your research, please consider citing:
```
@inproceedings{DBLP:conf/acl/HuCZJ23,
  author       = {Anwen Hu and
                  Shizhe Chen and
                  Liang Zhang and
                  Qin Jin},
  title        = {InfoMetIC: An Informative Metric for Reference-free Image Caption
                  Evaluation},
  booktitle    = {{ACL} {(1)}},
  pages        = {3171--3185},
  publisher    = {Association for Computational Linguistics},
  year         = {2023}
}
```



