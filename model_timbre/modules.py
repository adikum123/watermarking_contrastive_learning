from base64 import encode

import torch
import torch.nn as nn
from torch.nn import LeakyReLU

from .blocks import Conv1DBlock, FCBlock, Mish, PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=6, transformer_drop=0.1, attention_heads=8):
        super(Encoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=attention_heads, dropout=transformer_drop)
        self.dec_encoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=attention_heads, dropout=transformer_drop)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, nlayers_encoder)
        self.decoder = nn.TransformerDecoder(self.dec_encoder_layer, nlayers_encoder)

        #MLP for the input audio waveform
        self.wav_linear_in = FCBlock(win_dim, embedding_dim, activation=LeakyReLU(inplace=True))
        self.wav_linear_out = FCBlock(embedding_dim, win_dim)

        #MLP for the input wm
        self.msg_linear_in = FCBlock(msg_length, embedding_dim, activation=LeakyReLU(inplace=True))

        #position encoding
        self.pos_encoder = PositionalEncoding(d_model=embedding_dim, dropout=transformer_drop)


    def forward_encode_msg(self, x, w):
        x_embedding = self.wav_linear_in(x) # maps input audio waveform to embedding dimension
        p_x = self.pos_encoder(x_embedding) # applies positional encoding
        encoder_out = self.encoder(p_x.transpose(0,1)).transpose(0,1)   # gets the encoder output on positional encoded input [B, T, D]
        # Temporal Average Pooling
        wav_feature = torch.mean(encoder_out, dim=1, keepdim=True) # applies temporal average pooling [B, 1, D] gets a summary of the audio features
        msg_feature = self.msg_linear_in(w) # maps input wm [B, message_length] to embedding dimension [B, D]
        encoded_msg = wav_feature.add(msg_feature) # combines the audio feature and wm feature by addition [B, 1, D]
        return encoded_msg, encoder_out, p_x

    def forward_decode_wav(self, encoded_msg, encoder_out, p_x):
        # B, _, D = encoded_msg.shape
        encode_msg_repeat = encoded_msg.repeat(1, p_x.size(1), 1) # repeats the encoded message to match the time dimension of p_x [B, T, D]
        # applies the decoder on the sum of the repeated encoded message and positional encoding [B, T, D]
        embeded = self.decoder((encode_msg_repeat + p_x).transpose(0,1), memory=encoder_out.transpose(0,1)).transpose(0,1)
        wav_out = self.wav_linear_out(embeded) # applies a linear transformation to get the output waveform [B, T, D]
        return wav_out # returns the output waveform

    def forward(self, x, w):
        encoded_msg, encoder_out, p_x = self.forward_encode_msg(x, w)
        wav_out = self.forward_decode_wav(encoded_msg, encoder_out, p_x)
        return wav_out



class Decoder(nn.Module):
    def __init__(self, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=6, transformer_drop=0.1, attention_heads=8):
        super(Decoder, self).__init__()
        self.msg_decoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=attention_heads, dropout=transformer_drop)
        self.msg_decoder = nn.TransformerEncoder(self.msg_decoder_layer, nlayers_decoder)
        self.msg_linear_out = FCBlock(embedding_dim, msg_length)
        #MLP for the input audio waveform
        self.wav_linear_in = FCBlock(win_dim, embedding_dim, activation=LeakyReLU(inplace=True))
        #position encoding
        self.pos_encoder = PositionalEncoding(d_model=embedding_dim, dropout=transformer_drop)

    def forward(self, x):
        x_embedding = self.wav_linear_in(x) # converts raw waveform to embedding dimension
        p_x = self.pos_encoder(x_embedding) # applies positional encoding
        encoder_out = self.msg_decoder(p_x.transpose(0,1)).transpose(0,1) # gets the encoded output from the transformer decoder
        # Temporal Average Pooling to get features of watermarkes signal
        wav_feature = torch.mean(encoder_out, dim=1, keepdim=True)
        # Maps the pooled representation back to the predicted watermark message
        out_msg = self.msg_linear_out(wav_feature)
        return out_msg


class Discriminator(nn.Module):
    def __init__(self, msg_length, win_dim, embedding_dim, nlayers_decoder=6, transformer_drop=0.1, attention_heads=8):
        super(Decoder, self).__init__()
        self.msg_decoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=attention_heads, dropout=transformer_drop)
        self.msg_decoder = nn.TransformerEncoder(self.msg_decoder_layer, nlayers_decoder)
        self.msg_linear_out = FCBlock(embedding_dim, msg_length)
        #MLP for the input audio waveform
        self.wav_linear_in = FCBlock(win_dim, embedding_dim, activation=Mish())
        #position encoding
        self.pos_encoder = PositionalEncoding(d_model=embedding_dim, dropout=transformer_drop)

    def forward(self, x):
        x_embedding = self.wav_linear_in(x)
        p_x = self.pos_encoder(x_embedding)
        encoder_out = self.msg_decoder(p_x)
        # Temporal Average Pooling
        wav_feature = torch.mean(encoder_out, dim=1, keepdim=True) # [B, 1, H]
        out_msg = self.msg_linear_out(wav_feature)
        return torch.mean(out_msg)


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param