"""
CTSA:cated crossmodal transformer
拼接跨模态transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from models.subNets.transformers_encoder.transformer import TransformerEncoder

class TextSubnet(nn.Module):            #文本特征 预处理网络
    def __init__(self, args):
        super(TextSubnet, self).__init__()
        self.lstm=nn.LSTM(args.orig_d_l,hidden_size=args.text_hidden, num_layers=1, dropout=args.text_lstm_dropout, bidirectional=False,
                            batch_first=True)
    def forward(self,sequence):
        h1,(h_finall,c_finall)=self.lstm(sequence)
        return h1,h_finall
class VisionSubnet(nn.Module):          #视频特征 预处理网络
    def __init__(self, args):
        super(VisionSubnet, self).__init__()
        self.lstm = nn.LSTM(args.orig_d_v, hidden_size=args.vision_hidden, num_layers=1, dropout=args.vision_lstm_dropout, bidirectional=False,
                            batch_first=True)
        self.norm = nn.BatchNorm1d(args.orig_d_v)
    def forward(self,sequence,length):
        sequence = sequence.permute(0,2,1)
        sequence=self.norm(sequence)
        sequence = sequence.permute(0,2,1)
        packed_sequence = pack_padded_sequence(sequence, length, batch_first=True, enforce_sorted=False)
        packed_h1, (final_h1, final_c1)=self.lstm(packed_sequence)
        padded_h1, length = pad_packed_sequence(packed_h1)
        return padded_h1,torch.squeeze(final_c1)

class AudioSubnet(nn.Module):           #音频特征 预处理网络
    def __init__(self, args):
        super(AudioSubnet, self).__init__()
        self.lstm = nn.LSTM(args.orig_d_a, hidden_size=args.audio_hidden, num_layers=1, dropout=args.audio_lstm_dropout, bidirectional=False,
                            batch_first=True)
        self.nrom=torch.nn.LayerNorm(args.orig_d_a)
    def forward(self,sequence,length):
        packed_sequence = pack_padded_sequence(sequence, length, batch_first=True, enforce_sorted=False)
        packed_h1, (final_h1, final_c1)=self.lstm(packed_sequence)
        padded_h1, length = pad_packed_sequence(packed_h1)
        return padded_h1,final_c1


class Mult(nn.Module):
    def __init__(self, args):
        super(Mult, self).__init__()
        self.transformerencoder=TransformerEncoder(embed_dim=args.transformer_embed_dim,
                                  num_heads=args.num_heads,
                                  layers=args.layers,
                                  attn_dropout=args.attn_dropout,
                                  relu_dropout=0,
                                  res_dropout=0.1,
                                  embed_dropout=0.1,
                                  attn_mask=True)
    def forward(self,sequence_q,sequence_k):
        sequence_q=sequence_q.permute(1,0,2)
        sequence_k = sequence_k.permute(1, 0, 2)
        after_fuse = self.transformerencoder(sequence_q, sequence_k, sequence_k)
        return after_fuse.permute(1,0,2)


class CTSA(nn.Module):

    def __init__(self, args):
        super(CTSA, self).__init__()

        self.text_subnet=TextSubnet(args)
        self.vision_subnet=VisionSubnet(args)
        self.audio_subnet = AudioSubnet(args)

        self.linear_text_1=nn.Linear(args.transformer_embed_dim,args.transformer_embed_dim)
        self.linear_text_2=nn.Linear(args.transformer_embed_dim,20)
        self.linear_text_3=nn.Linear(20,3)

        self.linear_audio_1 = nn.Linear(args.transformer_embed_dim, args.transformer_embed_dim)
        self.linear_audio_2 = nn.Linear(args.transformer_embed_dim, 20)
        self.linear_audio_3 = nn.Linear(20, 3)

        self.linear_vision_1 = nn.Linear(args.transformer_embed_dim, args.transformer_embed_dim)
        self.linear_vision_2 = nn.Linear(args.transformer_embed_dim, 20)
        self.linear_vision_3 = nn.Linear(20, 3)

        self.fusenet_la=  Mult(args)
        self.fusenet_lv = Mult(args)
        self.fusenet_av = Mult(args)
        self.fusenet_al = Mult(args)
        self.fusenet_va = Mult(args)
        self.fusenet_vl = Mult(args)

        self.linear_output_1=nn.Linear(600,600)
        self.linear_output_2 = nn.Linear(600, 100)
        self.linear_output_3 = nn.Linear(100, 20)
        self.linear_output_4 = nn.Linear(20, 3)

    def forward(self, text_x, audio_x, video_x):

        self.text_x=text_x
        text_h,final_text=self.text_subnet(self.text_x)
        text_output_1=self.linear_text_2(self.linear_text_1(final_text))
        text_output_2=self.linear_text_3(text_output_1)
        text_output=torch.squeeze(text_output_2)

        self.vision_x=video_x[0]
        self.vision_x_length=video_x[1]
        vision_h,final_vision=self.vision_subnet(self.vision_x,self.vision_x_length)
        vision_output_1=self.linear_vision_2(self.linear_vision_1(final_vision))
        vision_output_2=self.linear_vision_3(vision_output_1)
        vision_output=torch.squeeze(vision_output_2)
        vision_h=vision_h.permute(1,0,2)

        self.audio_x=audio_x[0]
        self.audio_x_length=audio_x[1]
        audio_h,final_audio=self.audio_subnet(self.audio_x,self.audio_x_length)
        audio_output_1=self.linear_audio_2(self.linear_audio_1(final_audio))
        audio_output_2=self.linear_audio_3(audio_output_1)
        audio_output=torch.squeeze(audio_output_2)
        audio_h=audio_h.permute(1,0,2)

        fuse_la=self.fusenet_la(text_h,audio_h)[:,0,:]
        fuse_lv=self.fusenet_lv(text_h,vision_h)[:,0,:]
        fuse_av=self.fusenet_av(audio_h,vision_h)[:,0,:]
        fuse_al=self.fusenet_al(audio_h,text_h)[:,0,:]
        fuse_va=self.fusenet_va(vision_h,audio_h)[:,0,:]
        fuse_vl=self.fusenet_vl(vision_h,text_h)[:,0,:]
        fuse_all=torch.cat((fuse_la,fuse_lv,fuse_av,fuse_al,fuse_va,fuse_vl),dim=1)
        output=self.linear_output_4(self.linear_output_3(self.linear_output_2(self.linear_output_1(fuse_all))))

        result={'V':vision_output,
                'A':audio_output,
                'T':text_output,
                'M':output}
        return result

