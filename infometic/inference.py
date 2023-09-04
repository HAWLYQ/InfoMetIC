from model.ClipSeq import CLIPSeqModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import time
from tqdm import tqdm
import copy
import os
import lmdb
import numpy as np
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from clip_info.simple_tokenizer import SimpleTokenizer as _Tokenizer
import json
import argparse
import math
import io
from icecream import ic

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='COCO_val2014_000000197461.jpg', help='image name')
    parser.add_argument('--caption', type=str, default=' A very large sheep is standing under clouds.', help='caption')
    
    # inference setting
    parser.add_argument('--batch_size', type=int, default=500, help='the batch size of each gpu')
    # parser.add_argument('--checkpoint_path', type=str, default='save/clip-seq-global-scale_lr1e4_flickr30k/model_3.pth', help='checkpoint to inference')
    parser.add_argument('--checkpoint_path', type=str, default='save/infometic/best_model.pth', help='checkpoint to inference')
    parser.add_argument('--clip_score', action='store_true', help='calculate clip score, ignore proposed model.')
    parser.add_argument('--finetune_backbone', type=bool, default=False, help='whether to finetune CLIP backbone')
    parser.add_argument('--clip_model_name', type=str, default='ViT-B/32', choices=['ViT-B/32', 'ViT-B/16', 'RN50', 'RN101'])

    parser.add_argument('--image_input_type', type=str, default='feat', choices=['feat', 'finetuned_feat', 'cuts'])
    parser.add_argument('--image_length', type=int, default=100, help='the length of image region sequence')
    parser.add_argument('--bbox_cluster_num', type=int, default=20, help='the cluster num of bounding boxes')
    parser.add_argument('--image_label_type', type=str, default='covered', choices=['covered', 'uncovered'])
    parser.add_argument('--text_length', type=int, default=32, help='the length of bpe token sequence in CLIP(including sot and ent)')
    parser.add_argument('--bbox_text_length', type=int, default=10, help='the length of bpe token sequence of object word(including sot and ent)')

    # ii: vision recall score; tt: text precision score; it: global vision-text score
    parser.add_argument('--global_type', type=str, default='it-ii-tt', choices=['it','ii', 'tt','it-ii', 'it-tt', 'ii-tt', 'it-ii-tt'])
    parser.add_argument('--use_cross_att', type=bool, default=True, help='whether to use cross attention after multimodal transformer, set true when ii or tt in global')
    parser.add_argument('--keep_clip_global', type=bool, default=True, help='whether to fix clip global features, conflict with it global type')
    parser.add_argument('--individual_token_emb', type=bool, default=True, help='whether to individually embedding text tokens, when not keep_clip_global, this should be False')
    parser.add_argument('--use_global_img', type=bool, default=True, help='whether to use global image features during encoding')
    parser.add_argument('--add_pad_img_token', type=bool, default=True, help='whether to add a pad token after image feats')
    parser.add_argument('--use_bbox_position', type=bool, default=True, help='whether to use bounding box')
    parser.add_argument('--normalize_bbox_position', type=bool, default=True, help='whether to normalize bounding box')
    parser.add_argument('--use_bbox_text', type=bool, default=False, help='whether to use object labels of bounding boxes')
    parser.add_argument('--keep_clip_scale', type=bool, default=False, help='whether to keep scale as clip (scale.exp()==100.0)')
    
    parser.add_argument('--fusion_dim', type=int, default=512, help='the hidden size of fusion encoder')
    parser.add_argument('--inter_layers', type=int, default=2, help='the transformer layers of fusion encoder')
    parser.add_argument('--intra_layers', type=int, default=4, help='the transformer layers of fusion encoder')
    parser.add_argument('--fusion_heads', type=int, default=16, help='the head size of multi-head attention in fusion encoder')

    parser.add_argument('--multipy_with_scale', type=bool, default=True, help='whether to multiply cosine similarity with the scale')


    args = parser.parse_args()
    return args

def obj_to_cuda(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cuda()
    elif isinstance(obj, list):
        return [obj_to_cuda(t) for t in obj]
    elif isinstance(obj, dict):
        return {key: obj_to_cuda(t) for key, t in obj.items()}
    else:
        return obj


def get_image_input(imgid, image_feat, bbox_text_length, image_length, add_pad_img_token):
    dump = image_feat.get(str(imgid).encode('utf-8'))
    with io.BytesIO(dump) as reader:
        image_data = np.load(reader, allow_pickle=True)
        feat = image_data['feat']  # numpy.ndarry 100 * feat_dim
        bbox = image_data['bbox']  # numpy.ndarry 100 * 4
        bbox_text = image_data['bbox_text'][:, :bbox_text_length]  # numpy.ndarry 100 * 10 (bpe token ids of object word)
        # assert len(feat) >= image_length
        global_feat = image_data['global_feat']
        global_bbox = image_data['global_bbox']
        if len(feat) < image_length:
            # print('raw shape:', image.shape)
            image_mask = [1] * len(feat) + [0] * (image_length - len(feat))
            feat = np.pad(feat, ((0, image_length - len(feat)), (0, 0)), 'constant', constant_values=(0, 0))
            bbox = np.pad(bbox, ((0, image_length - len(bbox)), (0, 0)), 'constant', constant_values=(0, 0))
            bbox_text = np.pad(bbox_text, ((0, image_length - len(bbox_text)), (0, 0)), 'constant', constant_values=(0, 0))
            # print('padded image:', image)
            # print('image mask:', image_mask)
        else:
            feat = feat[:image_length]
            bbox = bbox[:image_length]
            bbox_text = bbox_text[:image_length]
            image_mask = [1] * image_length
        
        # using clustered bounding boxes rather than all bounding boxes
        clustered_bbox_index = image_data['clustered_bbox']
        if add_pad_img_token:
            # the last one in image sequence is pad token
            clustered_feat = np.zeros([len(clustered_bbox_index)+1, 512]) 
            clustered_bbox = np.zeros([len(clustered_bbox_index)+1, 4])
            clustered_bbox_text = np.zeros([len(clustered_bbox_index)+1, bbox_text_length], dtype=np.int)
            clustered_image_mask = np.ones(len(clustered_bbox_index)+1, dtype=np.int) # all feats in cluster should be attended
        else:
            clustered_feat = np.zeros([len(clustered_bbox_index), 512])
            clustered_bbox = np.zeros([len(clustered_bbox_index), 4])
            clustered_bbox_text = np.zeros([len(clustered_bbox_index), bbox_text_length], dtype=np.int)
            clustered_image_mask = np.ones(len(clustered_bbox_index), dtype=np.int) # all feats in cluster should be attended

        for i, raw_index in enumerate(clustered_bbox_index):
            clustered_feat[i] = feat[raw_index]
            clustered_bbox[i] = bbox[raw_index]
            clustered_bbox_text[i] = bbox_text[raw_index]
            # clustered_image_mask[i] = image_mask[raw_index]

    return clustered_feat, clustered_bbox, clustered_bbox_text, clustered_image_mask, global_feat, global_bbox

def process_text(caption, max_text_length, tokenizer):
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    text = [sot_token] + tokenizer.encode(caption) + [eot_token]
    text_length = len(text)
    if len(text) > max_text_length:
        text = text[:max_text_length]
        text[-1] = eot_token
        text_length = max_text_length
        # anwen hu 2021/11/14 mask start token and end token
        text_mask = [1] * max_text_length
        text_mask[0] = 0
        text_mask[-1] = 0
    else:
        # anwen hu 2021/11/14 mask start token and end token
        text_mask = [0] + [1] * (len(text)-2)+ [0] + [0] * (max_text_length - len(text))
        text = text + [0] * (max_text_length - len(text))
    text = torch.tensor(text)  # tensor
    text_mask = torch.tensor(text_mask)
    return text, text_length, text_mask

    
class MSCOCO_Infer():
    def __init__(self, opt, image_lmdb, fn2imgid_path):
        # read settings
        self.opt = opt
        assert self.opt.global_type == 'it-ii-tt'
        # load model
        self.model = CLIPSeqModel(self.opt)
        print('load model from', self.opt.checkpoint_path)
        self.model.load_state_dict(torch.load(self.opt.checkpoint_path))
        self.model.cuda()
        self.model.eval()
        # read preprocessd image feats (extacted by CLIP)
        feat_env = lmdb.open(image_lmdb, readonly=True, create=False)  # readahead=not _check_distributed()
        self.image_feats = feat_env.begin(buffers=True)
        # get the dict of image_name to imgid
        self.fn2imgid = json.load(open(fn2imgid_path, 'r', encoding='utf-8'))

        # determine the ratio of clipscore for infometic plus
        self.it_ratio = 100.0 / self.model.logit_scale.exp().item()
        ic(self.it_ratio)
    
    def evaluate(self, image_name, caption):
        ic(self.opt.global_type)
        ic(image_name)
        ic(caption)
        imgid = self.fn2imgid[image_name]
        clustered_feat, clustered_bbox, clustered_bbox_text, clustered_image_mask, global_feat, global_bbox = get_image_input(imgid,
                                                                                                                            self.image_feats,
                                                                                                                            self.opt.bbox_text_length,
                                                                                                                            self.opt.image_length,
                                                                                                                            self.opt.add_pad_img_token)
        # image input
        image = torch.tensor(clustered_feat).to(torch.float32).unsqueeze(0)  # tensor n_clusters * 512
        image_bbox = torch.tensor(clustered_bbox).to(torch.float32).unsqueeze(0)  # tensor n_clusters * 4
        bbox_text = torch.tensor(clustered_bbox_text).unsqueeze(0)  # tensor n_clusters * 10
        image_mask = torch.tensor(clustered_image_mask).unsqueeze(0) # tensor n_clusters
        image_global = torch.tensor(global_feat).unsqueeze(0)  # tensor 512
        bbox_global = torch.tensor(global_bbox).to(torch.float32).unsqueeze(0)  # tensor 

        tokenizer = _Tokenizer()
        # text input
        text, text_length, text_mask = process_text(caption, self.opt.text_length, tokenizer)
        bpe_token = tokenizer.decode(text.numpy()[:text_length].tolist(), return_list=True)
        text = text.unsqueeze(0)
        text_mask = text_mask.unsqueeze(0)

        with torch.no_grad():
            img_seq_scores, text_seq_scores, global_scores = self.model(image.cuda(), image_bbox.cuda(), bbox_text.cuda(), image_mask.cuda(), 
                                image_global.cuda(), bbox_global.cuda(), text.cuda(), text_mask.cuda(),
                                return_seq_pred_type = 'seq', # seq means return both fine-grained vision and text scores
                                return_clipscore=False, global_score_type=self.opt.global_type,
                                return_crossatt_score=True, multipy_with_scale=self.opt.multipy_with_scale)

            # combine multiple global type scores
            type2cores = {}
            global_types = self.opt.global_type.split('-')
            if len(global_types) == 1:
                assert not isinstance(global_scores, list)
                type2cores[global_types[0]] = global_scores
            else:
                assert isinstance(global_scores, list)
                for i in range(len(global_types)):
                    type2cores[global_types[i]] = global_scores[i]

            # fetch global scores
            infometic_recall = type2cores['ii']
            infometic_precision = type2cores['tt']
            infometic = infometic_recall + infometic_precision
            infometic_plus = infometic + self.it_ratio * type2cores['it']

            # fetch vision fine-grained scores
            vision_scores = img_seq_scores.squeeze(0).cpu().numpy().tolist()
            assert len(vision_scores) ==len(clustered_bbox)
            bbox_and_scores = [(clustered_bbox[i], round(vision_scores[i],4)) for i in range(len(clustered_bbox))]

            # fetch text fine-grained scores
            text_scores = text_seq_scores.squeeze(0).cpu().numpy().tolist()[:text_length]
            text_token_and_scores = [(bpe_token[i], round(text_scores[i],4)) for i in range(text_length)]
            
        return infometic_recall, infometic_precision, infometic, infometic_plus, bbox_and_scores, text_token_and_scores


if __name__ == '__main__':
    opt = parse_opt()
    image_name = opt.image
    caption = opt.caption

    # initialize the evaluator for mscoco image captioning
    infometic_mscoco = MSCOCO_Infer(opt=opt, image_lmdb='./data/mscoco_vit32_lmdb/', fn2imgid_path='./data/coco_fn2imgid.json')
    # perform evaluation
    infometic_r, infometic_p, infometic, infometic_plus, bbox_and_scores, text_token_and_scores= infometic_mscoco.evaluate(image_name=image_name, caption=caption)
    print('InfoMetIC Vision Recall Score:', infometic_r)
    print('InfoMetIC Text Precision Score:', infometic_p)
    print('InfoMetIC Score:', infometic)
    print('InfoMetIC+ Score:', infometic_plus)
    print('InfoMetIC fine-grained Vision Scores:', bbox_and_scores)
    print('InfoMetIC fine-grained Text Scores:', text_token_and_scores)

