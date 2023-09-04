import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
# print(sys.path)
import clip_info as clip
from clip_info.model import Transformer, LayerNorm
import numpy as np


class CLIPSeqModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # CLIP backbone
        if config.bbox_cluster_num != -1:
                self.image_length = config.bbox_cluster_num
        else:
            self.image_length = config.image_length
        
        if config.add_pad_img_token:
            self.image_length = self.image_length + 1
        
        # print(config.clip_model_name)
        self.clip_backbone, self.image_preprocess = clip.load(config.clip_model_name, device='cuda')
        
        self.use_global_img = config.use_global_img
        self.use_bbox_position = config.use_bbox_position
        self.use_bbox_text = config.use_bbox_text
        self.use_cross_att = config.use_cross_att
        self.keep_clip_global = config.keep_clip_global
        self.individual_token_emb = config.individual_token_emb
        self.normalize_bbox_position = config.normalize_bbox_position
        self.finetune_backbone = config.finetune_backbone
        self.image_input_type = config.image_input_type

        self.sot_id = 49406 # <|startoftext|>
        self.eot_id = 49407 # <|endoftext|>
        if not self.keep_clip_global:
            assert not self.individual_token_emb
        
        # inter-modal encoder
        self.fusion_encoder = Transformer(
            width=config.fusion_dim,
            layers=config.inter_layers,
            heads=config.fusion_heads
        )
        # vision intra-model encoder
        self.image_encoder = Transformer(
            width=config.fusion_dim,
            layers=config.intra_layers,
            heads=config.fusion_heads
        )
        # text intra-model encoder
        self.text_encoder = Transformer(
            width=config.fusion_dim,
            layers=config.intra_layers,
            heads=config.fusion_heads
        )

        self.ln = LayerNorm(config.fusion_dim)
        self.fusion_dim = config.fusion_dim

        if self.use_bbox_position:
            self.bbox_position_embedding = nn.Parameter(torch.empty(4, config.fusion_dim))
        if self.use_bbox_text:
            self.img_bboxtext_fusion = nn.Parameter(torch.empty(2*config.fusion_dim, config.fusion_dim))
        
        if config.keep_clip_scale:
            self.logit_scale = torch.ones([]) * np.log(1 / 0.01) # logit_scale.exp() == 100.0
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.use_cross_att:
            self.img_att_query = nn.Parameter(torch.empty(config.fusion_dim, config.fusion_dim))
            self.img_att_key = nn.Parameter(torch.empty(config.fusion_dim, config.fusion_dim))
            self.text_att_query = nn.Parameter(torch.empty(config.fusion_dim, config.fusion_dim))
            self.text_att_key = nn.Parameter(torch.empty(config.fusion_dim, config.fusion_dim))
    

        self.initialize_parameters()

    
    def initialize_parameters(self):
        # initialize fusion encoder
        proj_std = (self.fusion_encoder.width ** -0.5) * ((2 * self.fusion_encoder.layers) ** -0.5)
        attn_std = self.fusion_encoder.width ** -0.5
        fc_std = (2 * self.fusion_encoder.width) ** -0.5
        for block in self.fusion_encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # initialize intra encoders 
        proj_std = (self.image_encoder.width ** -0.5) * ((2 * self.image_encoder.layers) ** -0.5)
        attn_std = self.image_encoder.width ** -0.5
        fc_std = (2 * self.image_encoder.width) ** -0.5
        for block in self.image_encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        # text encoder shares the width and layers setting with image encoder
        for block in self.text_encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
    
        if self.use_bbox_position:
            nn.init.normal_(self.bbox_position_embedding, std=0.01)
        if self.use_bbox_text:
            nn.init.normal_(self.img_bboxtext_fusion, std=(2*self.fusion_dim) ** -0.5)
        if self.use_cross_att:
            nn.init.normal_(self.img_att_query, std=self.fusion_dim ** -0.5)
            nn.init.normal_(self.img_att_key, std=self.fusion_dim ** -0.5)
            nn.init.normal_(self.text_att_query, std=self.fusion_dim ** -0.5)
            nn.init.normal_(self.text_att_key, std=self.fusion_dim ** -0.5)
        


    def fusion_encoding(self, image_input, text_input):
        fusion_input = torch.cat([image_input, text_input], dim=1)  # batch * （img_seq+text_seq) * hid_dim
        fusion_input = fusion_input.permute(1, 0, 2)  # （img_seq+text_seq) * batch * hid_dim
        fusion_input = fusion_input.to(dtype=next(self.fusion_encoder.parameters()).dtype)
        fusion_output = self.fusion_encoder(fusion_input)  # （img_seq+text_seq) * batch * hid_dim
        fusion_output = fusion_output.permute(1, 0, 2)  # batch * （img_seq+text_seq) * hid_dim
        fusion_output = self.ln(fusion_output)  # batch * （img_seq+text_seq) * hid_dim
        return fusion_output
    
    def seperate_encoding(self, image_input, text_input):
        image_input = image_input.permute(1, 0, 2)  # img_seq * batch * hid_dim
        image_input = image_input.to(dtype=next(self.image_encoder.parameters()).dtype)
        image_output = self.image_encoder(image_input)  #  img_seq * batch * hid_dim
        image_output = image_output.permute(1, 0, 2)  # batch * img_seq * hid_dim
        image_output = self.ln(image_output)  # batch * img_seq * hid_dim

        text_input = text_input.permute(1, 0, 2)  # text_seq * batch * hid_dim
        text_input = text_input.to(dtype=next(self.text_encoder.parameters()).dtype)
        text_output = self.text_encoder(text_input)  #  text_seq * batch * hid_dim
        text_output = text_output.permute(1, 0, 2)  # batch * text_seq * hid_dim
        text_output = self.ln(text_output)  # batch * text_seq * hid_dim
        
        return image_output, text_output

    
    def global_score(self, img_global, img_seq, img_mask, text_global, text_seq, text_mask, global_score_type, return_crossatt_score, multipy_with_scale):
        assert global_score_type in ['it', 'ii', 'tt']
        _, img_len, _ = img_seq.size()
        _, text_len, _ = text_seq.size()
        if global_score_type == 'it': # image-text
            # normalize
            img_global = img_global / img_global.norm(dim=-1, keepdim=True)
            text_global = text_global / text_global.norm(dim=-1, keepdim=True)
            if multipy_with_scale:
                global_score = self.logit_scale.exp() * torch.sum(img_global * text_global, dim=-1) # batch
            else:
                global_score = torch.sum(img_global * text_global, dim=-1) # batch
        elif global_score_type  == 'tt': # text-text
            # using img_global as query, text_seq as key and value
            query = img_global @ self.text_att_query
            key = text_seq @ self.text_att_key
            text_seq_score = torch.sum(query.unsqueeze(1).repeat(1, text_len, 1) * key, dim=-1)  + (1 - text_mask) * -10000.0 # batch * text_seq
            text_seq_score = F.softmax(text_seq_score, dim=-1)
            attend_text = torch.sum(text_seq * text_seq_score.unsqueeze(-1).repeat(1, 1, self.fusion_dim), dim=1)  # batch * hid_dim
            # normalize
            attend_text = attend_text / attend_text.norm(dim=-1, keepdim=True)
            text_global = text_global / text_global.norm(dim=-1, keepdim=True)
            if multipy_with_scale:
                global_score = self.logit_scale.exp() * torch.sum(attend_text * text_global, dim=-1) # batch
            else:
                global_score = torch.sum(attend_text * text_global, dim=-1) # batch

            

        elif global_score_type  == 'ii': # image-image
            # using text_global as query, img_seq as key and value
            query = text_global @ self.img_att_query
            key = img_seq @ self.img_att_key
            img_seq_score = torch.sum(query.unsqueeze(1).repeat(1, img_len, 1) * key, dim=-1) + (1 - img_mask) * -10000.0 # batch * img_seq
            # img_seq_score = torch.sigmoid(img_seq_score) 
            img_seq_score = F.softmax(img_seq_score, dim=-1) 
            attend_img = torch.sum(img_seq * img_seq_score.unsqueeze(-1).repeat(1, 1, self.fusion_dim), dim=1)  # batch * hid_dim
            # normalize
            attend_img = attend_img / attend_img.norm(dim=-1, keepdim=True)
            img_global = img_global / img_global.norm(dim=-1, keepdim=True)
            if multipy_with_scale:
                global_score = self.logit_scale.exp() * torch.sum(img_global * attend_img, dim=-1) # batch
            else:
                global_score = torch.sum(img_global * attend_img, dim=-1) # batch

        if return_crossatt_score and global_score_type  == 'tt':  
            return global_score, text_seq_score # batch, batch * text_seq
        elif return_crossatt_score and global_score_type  == 'ii':
            return global_score, img_seq_score
        else:
            return global_score, None # batch, None

    def zeroshot_global_score(self, img_global, img_seq, img_mask, text_global, text_seq, text_mask, global_score_type, return_crossatt_score, multipy_with_scale):
        # print(global_score_type)
        assert global_score_type in ['it', 'ii', 'tt']
        _, img_len, _ = img_seq.size()
        _, text_len, _ = text_seq.size()
        if global_score_type == 'it': # image-text
            # normalize
            img_global = img_global / img_global.norm(dim=-1, keepdim=True)
            text_global = text_global / text_global.norm(dim=-1, keepdim=True)
            if multipy_with_scale:
                global_score = self.clip_backbone.logit_scale.exp() * torch.sum(img_global * text_global, dim=-1) # batch
            else:
                global_score = torch.sum(img_global * text_global, dim=-1) # batch
        elif global_score_type  == 'tt': # text-text
            # using img_global as query, text_seq as key and value
            query = img_global
            key = text_seq
            text_seq_score = torch.sum(query.unsqueeze(1).repeat(1, text_len, 1) * key, dim=-1)  + (1 - text_mask) * -10000.0 # batch * text_seq
            text_seq_score = F.softmax(text_seq_score, dim=-1)
            attend_text = torch.sum(text_seq * text_seq_score.unsqueeze(-1).repeat(1, 1, self.fusion_dim), dim=1)  # batch * hid_dim
            # normalize
            attend_text = attend_text / attend_text.norm(dim=-1, keepdim=True)
            text_global = text_global / text_global.norm(dim=-1, keepdim=True)
            if multipy_with_scale:
                global_score = self.clip_backbone.logit_scale.exp() * torch.sum(attend_text * text_global, dim=-1) # batch
            else:
                global_score = torch.sum(attend_text * text_global, dim=-1) # batch

        elif global_score_type  == 'ii': # image-image
            # using text_global as query, img_seq as key and value
            query = text_global
            key = img_seq
            img_seq_score = torch.sum(query.unsqueeze(1).repeat(1, img_len, 1) * key, dim=-1) + (1 - img_mask) * -10000.0 # batch * img_seq
            # img_seq_score = torch.sigmoid(img_seq_score) 
            img_seq_score = F.softmax(img_seq_score, dim=-1) 
            attend_img = torch.sum(img_seq * img_seq_score.unsqueeze(-1).repeat(1, 1, self.fusion_dim), dim=1)  # batch * hid_dim
            # normalize
            attend_img = attend_img / attend_img.norm(dim=-1, keepdim=True)
            img_global = img_global / img_global.norm(dim=-1, keepdim=True)
            if multipy_with_scale:
                global_score = self.clip_backbone.logit_scale.exp() * torch.sum(img_global * attend_img, dim=-1) # batch
            else:
                global_score = torch.sum(img_global * attend_img, dim=-1) # batch

        if return_crossatt_score and global_score_type  == 'tt':  
            return global_score, text_seq_score # batch, batch * text_seq
        elif return_crossatt_score and global_score_type  == 'ii':
            return global_score, img_seq_score
        else:
            return global_score, None # batch, None

    def zero_infoeval(self, clip_img_global, clip_img_feats, image_mask, clip_text_global, clip_text_feats, text_mask, 
                        global_score_type, return_crossatt_score, multipy_with_scale, return_global_score, return_seq_pred_type):
        if '-' not in global_score_type:
            seq_score_dict = {}
            global_score, seq_score = self.zeroshot_global_score(clip_img_global, clip_img_feats, image_mask, clip_text_global, clip_text_feats, text_mask, 
                                            global_score_type, return_crossatt_score, multipy_with_scale)
            seq_score_dict[global_score_type] = seq_score
        else:
            global_score = []
            seq_score_dict = {}
            for sub_global_score_type in global_score_type.split('-'):
                sub_global_score, sub_seq_score = self.zeroshot_global_score(clip_img_global, clip_img_feats, image_mask, clip_text_global, clip_text_feats, text_mask, 
                                            sub_global_score_type, return_crossatt_score, multipy_with_scale)
                global_score.append(sub_global_score)
                seq_score_dict[sub_global_score_type]=sub_seq_score 
        return global_score, seq_score_dict


    def backbone_encoding(self, image, bbox_text, image_global, text):
        # image encoding
        if self.image_input_type != 'cuts':
            clip_img_feats = image
            clip_img_global = image_global.to(torch.float32)
        else:
            clip_img_feats = self.clip_backbone.encode_image(image.view(-1, 3, 224, 224), patch_level=False) # (batch*img_seq) * hid_dim
            clip_img_feats = clip_img_feats.view(image.size(0), image.size(1), -1) # batch * (img_seq+1) * hid_dim 
            clip_img_global = self.clip_backbone.encode_image(image_global, patch_level=False).to(torch.float32) # batch * hid_dim
        # print('global feat:', clip_img_global)
        # exit(0)
        if self.use_bbox_text:
            bbox_text = bbox_text.reshape(-1, bbox_text.size(-1))  # (batch * img_seq) * 10
            clip_bboxtext_feats = self.clip_backbone.encode_text(bbox_text, token_level=False).to(torch.float32)  # (batch * img_seq) * hid_dim
            clip_bboxtext_feats = clip_bboxtext_feats.reshape(-1, self.image_length, self.fusion_dim) # batch * img_seq * hid_dim
        else:
            clip_bboxtext_feats = None

        # text encoding
        if self.individual_token_emb:
            token_text = text.view(-1, 1) # (batch*text_seq) * 1
            # add sot token and eot token to form a pseudo sentence, such as '<sot> man <eot>'
            token_text = F.pad(token_text, pad=(1,0,0,0), mode='constant', value=self.sot_id) 
            token_text = F.pad(token_text, pad=(0,1,0,0), mode='constant', value=self.eot_id) # (batch*text_seq) * 3
            # token-level features, but because of individually embedding, there are no global feat
            clip_text_feats = self.clip_backbone.encode_text(token_text, token_level=False).view(text.size(0), text.size(1), -1) # batch * text_seq * hid_dim
            # global text feat
            clip_text_global = self.clip_backbone.encode_text(text, token_level=False).to(torch.float32) # batch * hid_dim
        else:
            # token-level text features, <eot> token represent the global feat
            clip_text_feats = self.clip_backbone.encode_text(text, token_level=True)  # batch * text_seq * hid_dim
            clip_text_global = clip_text_feats[torch.arange(text.shape[0]), text.argmax(dim=-1)].to(torch.float32)

        return clip_img_feats, clip_img_global, clip_text_feats, clip_text_global, clip_bboxtext_feats

    def forward(self, image, image_bbox, bbox_text, image_mask, image_global, bbox_global, text, text_mask=None, 
                return_seq_pred_type=None, return_clipscore=False, global_score_type='it', 
                return_crossatt_score=False, return_global_score=True, multipy_with_scale=True,
                clip_zeroshot_infoeval=False):
        """
        :param image: batch * img_seq * hid_dim or batch * img_seq * 3 * 224 * 224
        :param image_bbox: batch * img_seq * 4
        :param bbox_text: batch * img_seq * text_seq
        :param image_mask: batch * img_seq (mask global image)
        :param image_global: batch * hid_dim or batch * 3 * 224 * 224
        :param bbox_global: batch * 4 (bbox of the whole image)
        :param text: batch * text_seq
        :param text_mask: batch * text_seq (mask padding tokens, start token and end token)

        :param return_clipscore: whether return clipscore
        :param global_score_type: 'it', 'ii', 'tt', 'it-ii', 'it-tt', 'ii-tt', 'it-ii-tt' 
        :param return_crossatt_score: whether to return token-level cross attion score, designed for ii or tt 
        :param return_global_score: whether to return global score
        :param multipy_with_scale: whether return the cosine similarity or multiplied with the scale
        :param clip_zeroshot_infoeval: whether to do informative evaluation without extra architectures
        :return:
        """

        assert global_score_type in ['it', 'ii', 'tt', 'it-ii', 'it-tt', 'ii-tt', 'it-ii-tt' ]
        # get clip image and text features
       
        if self.finetune_backbone:
            clip_img_feats, clip_img_global, clip_text_feats, clip_text_global, clip_bboxtext_feats = self.backbone_encoding(image, bbox_text, image_global, text)
        else:
            with torch.no_grad():
                clip_img_feats, clip_img_global, clip_text_feats, clip_text_global, clip_bboxtext_feats = self.backbone_encoding(image, bbox_text, image_global, text)
        
        if self.use_global_img:
            clip_img_feats = torch.cat([clip_img_global.unsqueeze(1), clip_img_feats], dim=1)  # batch * (img_seq+1) * hid_dim
            image_bbox = torch.cat([bbox_global.unsqueeze(1), image_bbox], dim=1)  # batch * (img_seq+1) * 4
            # mask global image during fine-grained scoring
            image_mask = F.pad(image_mask, pad=(1,0,0,0), mode='constant', value=0) # batch * (img_seq+1) 

        if return_clipscore:
            similarity = torch.cosine_similarity(clip_img_global, clip_text_global, dim=1).unsqueeze(1)  # batch * 1
            similarity, _ = torch.max(torch.cat([similarity, torch.zeros_like(similarity)], dim=-1), dim=-1)
            clipscore = 2.5 * similarity
            return clipscore
        
        if clip_zeroshot_infoeval:
            # directly return zeroshot evaluation
            global_score, seq_score_dict = self.zero_infoeval(clip_img_global, clip_img_feats, image_mask, clip_text_global, clip_text_feats, text_mask, 
                        global_score_type, return_crossatt_score, multipy_with_scale, return_global_score, return_seq_pred_type)
            if return_crossatt_score:
                if self.use_global_img:
                    img_preds = seq_score_dict['ii'][:, 1:]
                else:
                    img_preds = seq_score_dict['ii']
                text_preds = seq_score_dict['tt']
    
            # only return token-level score
            if return_seq_pred_type is not None and not return_global_score:
                if return_seq_pred_type == 'seq':
                    return img_preds, text_preds
                elif return_seq_pred_type == 'seq-image':
                    return img_preds
                else:
                    assert return_seq_pred_type == 'seq-text'
                    return text_preds
            # only return global score
            elif return_seq_pred_type is None and return_global_score:
                return global_score
            # return token-level and global score
            else:
                if return_seq_pred_type == 'seq':
                    return img_preds, text_preds, global_score
                elif return_seq_pred_type == 'seq-image':
                    return img_preds, global_score
                else:
                    assert return_seq_pred_type == 'seq-text'
                    return text_preds, global_score
        
        if self.use_bbox_position:
            if self.normalize_bbox_position:
                image_bbox = image_bbox / torch.cat([bbox_global[:,2:], bbox_global[:,2:]], dim=1).unsqueeze(1)
            clip_img_feats = clip_img_feats + image_bbox @ self.bbox_position_embedding  # batch * img_seq * hid_dim
        if self.use_bbox_text:
            clip_img_feats = torch.cat([clip_img_feats, clip_bboxtext_feats], dim=-1) @ self.img_bboxtext_fusion  # batch * img_seq * hid_dim
        
        batch_size, img_len, _ = clip_img_feats.size()
        _, text_len, _ = clip_text_feats.size()
        img_begin = 0
        img_end = img_len

        # vision and text intra-modal encoding
        img_output, text_output = self.seperate_encoding(clip_img_feats, clip_text_feats) 
        # inter-modal encoding
        fusion_output = self.fusion_encoding(img_output, text_output)
        img_output = fusion_output[:, img_begin:img_end, :] # batch * img_seq * hid_dim
        text_output = fusion_output[:, img_end:, :] # batch * text_seq * hid_dim
       
       
        # global score and token-level cross-att score
        if not self.keep_clip_global:
            assert not self.individual_token_emb
            img_global_output = img_output[:, 0, :]  # batch * 512
            text_global_output = text_output[torch.arange(text.shape[0]), text.argmax(dim=-1)]  # fetch eot token output, like clip, batch * 512
        else:
            img_global_output = clip_img_global
            text_global_output = clip_text_global
        if '-' not in global_score_type:
            seq_score_dict = {}
            global_score, seq_score = self.global_score(img_global_output, img_output, image_mask, text_global_output, text_output, text_mask, 
                                            global_score_type, return_crossatt_score, multipy_with_scale)
            seq_score_dict[global_score_type] = seq_score
            # return global_score
            # return global_score, seq_score_dict
        else:
            global_score = []
            seq_score_dict = {}
            for sub_global_score_type in global_score_type.split('-'):
                sub_global_score, sub_seq_score = self.global_score(img_global_output, img_output, image_mask, text_global_output, text_output, text_mask, 
                                                        sub_global_score_type, return_crossatt_score, multipy_with_scale)
                global_score.append(sub_global_score)
                seq_score_dict[sub_global_score_type]=sub_seq_score 
            # return  global_score
            # return global_score, seq_score_dict
        
        # fetch fine-grained scores
        if return_seq_pred_type == 'seq':
            if self.use_global_img:
                img_preds = seq_score_dict['ii'][:, 1:]
                text_preds = seq_score_dict['tt']
            else:
                img_preds = seq_score_dict['ii']
                text_preds = seq_score_dict['tt']
        elif return_seq_pred_type == 'seq-image':
            if self.use_global_img:
                img_preds = seq_score_dict['ii'][:, 1:]
            else:
                img_preds = seq_score_dict['ii']
        elif return_seq_pred_type == 'seq-text':
            text_preds = seq_score_dict['tt']
        
        # only return token-level score
        if return_seq_pred_type is not None and not return_global_score:
            if return_seq_pred_type == 'seq':
                return img_preds, text_preds
            elif return_seq_pred_type == 'seq-image':
                return img_preds
            else:
                assert return_seq_pred_type == 'seq-text'
                return text_preds
        # only return global score
        elif return_seq_pred_type is None and return_global_score:
            return global_score
        # return token-level and global score
        else:
            if return_seq_pred_type == 'seq':
                return img_preds, text_preds, global_score
            elif return_seq_pred_type == 'seq-image':
                return img_preds, global_score
            else:
                assert return_seq_pred_type == 'seq-text'
                return text_preds, global_score

            

        
    
    









