import torch
from torchvision import models
from torch import nn
from efficientnet.model import EfficientNet

class CNN_Encoder(nn.Module):
    def __init__(self, class_n):
        super(CNN_Encoder, self).__init__()
        # self.model = models.resnet50(pretrained=True)
        self.model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=class_n)
    
    def forward(self, inputs):
        output = self.model(inputs)
        return output

# class RNN_Decoder(nn.Module):
#     def __init__(self, max_len, embedding_dim, num_features, class_n, rate):
#         super(RNN_Decoder, self).__init__()
#         self.lstm = nn.LSTM(max_len, embedding_dim)
#         self.rnn_fc = nn.Linear(num_features*embedding_dim, 1000)
#         self.final_layer1 = nn.Linear(1000 + 1000, 128) # efficientnet out_dim + lstm out_dim
#         self.final_layer2 = nn.Linear(128, class_n)
#         self.dropout = nn.Dropout(rate)
#         self.dropout2 = nn.Dropout(0.5)

#     def forward(self, enc_out, dec_inp):
#         hidden, _ = self.lstm(dec_inp)
#         hidden = hidden.view(hidden.size(0), -1)
#         hidden = self.rnn_fc(hidden)
#         concat = torch.cat([enc_out, hidden], dim=1) # enc_out + hidden 
#         fc_input = concat
#         fc_input = self.dropout2(self.final_layer1(fc_input))
#         output = self.dropout((self.final_layer2(fc_input)))
#         return output

class RNN_Decoder(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, class_n, rate):
        super(RNN_Decoder, self).__init__()
        self.lstm = nn.LSTM(max_len, embedding_dim)
        self.rnn_fc = nn.Linear(num_features*embedding_dim, 1000)
        self.final_layer = nn.Linear(1000 + 1000, class_n) # efficientnet out_dim + lstm out_dim
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_out, dec_inp):
        hidden, _ = self.lstm(dec_inp)
        hidden = hidden.view(hidden.size(0), -1)
        hidden = self.rnn_fc(hidden)
        concat = torch.cat([enc_out, hidden], dim=1) # enc_out + hidden 
        fc_input = concat
        output = self.dropout((self.final_layer(fc_input)))
        return output

class CNN2RNN(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, class_n, rate):
        super(CNN2RNN, self).__init__()
        self.cnn = CNN_Encoder(class_n)
        self.rnn = RNN_Decoder(max_len, embedding_dim, num_features, class_n, rate)
        
    def forward(self, img, seq):
        cnn_output = self.cnn(img)
        output = self.rnn(cnn_output, seq)
        return output