"STSA模型的单任务模式"


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
        self.norm = nn.BatchNorm1d(args.orig_d_a)
    def forward(self,sequence,length):
        sequence = sequence.permute(0,2,1)
        sequence=self.norm(sequence)
        sequence = sequence.permute(0,2,1)
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
    def forward(self,sequence_q,sequence_k,gate_q=None,gate_k=None):
        sequence_q=sequence_q.permute(1,0,2)
        sequence_k = sequence_k.permute(1, 0, 2)
        if gate_q==None:
            after_fuse = self.transformerencoder(sequence_q, sequence_k, sequence_k)
        else:
            after_fuse = self.transformerencoder(sequence_q, sequence_k, sequence_k, gate_q, gate_k)
        return after_fuse.permute(1,0,2)


class STSA_S(nn.Module):

    def __init__(self, args):
        super(STSA_S, self).__init__()

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

        # self.linear_fuse_1 = nn.Linear(args.transformer_embed_dim,args.transformer_embed_dim)
        # self.linear_fuse_2 = nn.Linear(args.transformer_embed_dim,20)
        # self.linear_fuse_3 = nn.Linear(20, 3)

        self.linear_fuse_audio_1=nn.Linear(args.transformer_embed_dim,args.transformer_embed_dim)
        self.linear_fuse_audio_2 = nn.Linear(args.transformer_embed_dim, 20)
        self.linear_fuse_audio_3 = nn.Linear(20, 3)

        self.fusenet1=  Mult(args)
        self.fusenet2 = Mult(args)

        self.linear_f_1=nn.Linear(100,100)
        self.linear_f_2 = nn.Linear(100, 20)
        self.linear_f_3 = nn.Linear(20, 3)

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


        fuse_1=self.fusenet1(text_h,audio_h)


        audio_fuse_output_1=self.linear_fuse_audio_2(self.linear_fuse_audio_1(fuse_1[:,0,:]))
        audio_fuse_output=self.linear_fuse_audio_3(audio_fuse_output_1)

        fuse_2=self.fusenet2(fuse_1,vision_h)

        # t=torch.squeeze(text_output_1)
        # a=torch.squeeze(audio_fuse_output_1)
        # v=vision_output_1
        # y=torch.squeeze(audio_output_1)
        t=torch.squeeze(text_output_1)
        a=torch.squeeze(audio_fuse_output_1)
        all=fuse_2[:,0,:]
        # final_all=torch.cat((t,v,all),dim=1)
        fuse_output = self.linear_f_3(self.linear_f_2(self.linear_f_1(all)))


        result={
                'M':fuse_output
                }
        return result
