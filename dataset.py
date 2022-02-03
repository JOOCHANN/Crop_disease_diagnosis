import torchvision.transforms as transforms
import pandas as pd
import json
import torch
from torch.utils.data.dataset import Dataset
from augmentation import RandAugment
from glob import glob
from PIL import Image

from utils import *

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

class CROP(Dataset):
    def __init__(self, train_data_path, mode='train'):
        super(CROP, self).__init__()

        self.files = sorted(glob(train_data_path + '/*'))
        self.mode = mode
        csv_files = sorted(glob(train_data_path + '/*/*.csv'))
        # self.csv_feature_dict = data_min_max(csv_files)
        self.label_encoder, self.label_decoder = make_label_encoder_decoder()
        self.csv_feature_check = [0]*len(self.files)
        self.csv_features = [None]*len(self.files)
        self.max_len = 24 * 6 # 하루

        # 시간 걸려서 일단 고정
        self.csv_feature_dict = {'내부 온도 1 평균': [3.4, 47.3], '내부 온도 1 최고': [3.4, 47.6], '내부 온도 1 최저': [3.3, 47.0], '내부 습도 1 평균': [23.7, 100.0], '내부 습도 1 최고': [25.9, 100.0], '내부 습도 1 최저': [0.0, 100.0], '내부 이슬점 평균': [0.1, 34.5], '내부 이슬점 최고': [0.2, 34.7], '내부 이슬점 최저': [0.0, 34.4]}

    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split('/')[-1].split('\\')[-1]

        # csv
        if self.csv_feature_check[i] == 0:
            csv_path = f'{file}/{file_name}.csv'
            df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
            df = df.replace('-', 0)

            # MinMax scaling
            for col in df.columns:
                df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
                df[col] = df[col] / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])

            # zero padding
            pad = np.zeros((self.max_len, len(df.columns)))
            length = min(self.max_len, len(df))
            pad[-length:] = df.to_numpy()[-length:]

            # transpose to sequential data
            csv_feature = pad.T
            self.csv_features[i] = csv_feature
            self.csv_feature_check[i] = 1
        else:
            csv_feature = self.csv_features[i]
        
        # image
        image_path = f'{file}/{file_name}.jpg'
        img = Image.open(image_path)

        # Data Augmentation
        if self.mode == 'train':
            trans = transforms.Compose([
                transforms.Resize((600, 600), interpolation=Image.BICUBIC),
                transforms.CenterCrop((500, 500)),
                RandAugment(2, 14),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            trans = transforms.Compose([
                transforms.Resize((600, 600), interpolation=Image.BICUBIC),
                transforms.CenterCrop((500, 500)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            # trans2 = transforms.Compose([
            #     transforms.Resize((600, 600), interpolation=Image.BICUBIC),
            #     transforms.CenterCrop((500, 500)),
            #     transforms.RandomHorizontalFlip(1),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean, std)
            # ])

        img1 = trans(img)
        # img2 = trans(img)

        if self.mode == 'train':
            json_path = f'{file}/{file_name}.json'
            with open(json_path, 'r') as f:
                json_file = json.load(f)
            
            crop = json_file['annotations']['crop']
            disease = json_file['annotations']['disease']
            risk = json_file['annotations']['risk']
            label = f'{crop}_{disease}_{risk}'

            return {
                'img' : img1,
                'csv_feature' : torch.tensor(csv_feature, dtype=torch.float32),
                'label' : torch.tensor(self.label_encoder[label], dtype=torch.long)
            }
        else:
            return {
                'img' : img1,
                # 'img2' : img2,
                'csv_feature' : torch.tensor(csv_feature, dtype=torch.float32),
            }

    def __len__(self):
        return len(self.files)