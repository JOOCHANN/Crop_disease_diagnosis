import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import argparse
import torch
from efficientnet.model import EfficientNet
from torch.utils.data import DataLoader
from utils import *
from dataset import CROP
from model import *

l_e, l_d = make_label_encoder_decoder()

SEED = "0"
BATCH_SIZE = 64
NUM_CLASS = len(l_e)
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 512
NUM_FEATURES = 9
MAX_LEN = 24*6
DROPOUT_RATE = 0.1
EPOCHS = 29
STEP_SIZE = 10

class_weights = torch.tensor([0.28479012,  1.61314685,  1.2205291,   0.1959898,   1.49792208,  2.0781982,
                                5.49238095,  1.38963855,  1.62450704,  1.47871795,  1.50771242,  0.25155943,
                                3.34318841,  2.33010101,  1.55864865,  1.45081761,  1.46929936,  0.27859903,
                                5.767,      19.22333333, 17.74461538,  7.95448276, 12.81555556, 16.47714286,
                                10.9847619])

def main(mode):
    # SEED
    # torch.manual_seed(SEED)
    use_gpu = torch.cuda.is_available()
    print("Use gpu :", use_gpu)
    cudnn.benchmark = True
    # torch.cuda.manual_seed_all(SEED)

    # data loading
    if mode =='train':
        print("Train dataset loading")
        trainset = CROP(train_data_path, mode=mode)
        trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        print("Test dataset loading")
        testset = CROP(test_data_path, mode='test')
        testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN2RNN(max_len=MAX_LEN, embedding_dim=EMBEDDING_DIM, num_features=NUM_FEATURES, class_n=NUM_CLASS, rate=DROPOUT_RATE)
    model = model.to("cuda") # single gpu
    # model = nn.DataParallel(model).cuda() # multi gpu

    if mode =='test':
        model.load_state_dict(torch.load(load_model_path))
        model = model.to("cuda")
        model.eval()
        results = []

        for batch, batch_item in enumerate(testloader):
            img1 = batch_item['img'].to('cuda')
            # img2 = batch_item['img2'].to('cuda')
            seq = batch_item['csv_feature'].to('cuda')
            with torch.no_grad():
                output = model(img1, seq)
                # output2 = model(img2, seq)
            
            # output1 = output1.unsqueeze(2)
            # output2 = output2.unsqueeze(2)
            # output = torch.cat([output1, output2], dim=2)
            # output = torch.max(output, 2)[0]
            output = torch.tensor(torch.argmax(output, dim=1), dtype=torch.int32).cpu().numpy()
            results.extend(output)

            if (batch+1) % 200 == 0:
                print("==> {}/{}".format(batch+1, len(testloader)))
            
        preds = np.array([l_d[int(val)] for val in results])
        submission = pd.read_csv('data/sample_submission.csv')
        submission['label'] = preds
        submission.to_csv(out_file, index=False)
        exit(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(weight=class_weights.to("cuda"))
    # criterion = FocalLoss()

    # 매 stepsize마다 learning rate를 0.1씩 감소하는 scheduler 생성
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.1)

    loss_plot = []
    metric_plot = []

    for epoch in range(EPOCHS):
        total_loss = 0
        total_acc = 0

        print("==> Epoch {}/{}".format(epoch+1, EPOCHS))

        training = True
        for batch, batch_item in enumerate(trainloader):
            batch_loss, batch_acc = train_step(model, optimizer, criterion, batch_item, training)
            total_loss += batch_loss
            total_acc += batch_acc

            if (batch+1) % 100 == 0:
                print("Batch {}/{}\t Loss {:.6f}\t Mean Loss {:.6f}\t Mean F-1 {:.6f}" \
                    .format(batch+1, len(trainloader), batch_loss.item(), total_loss/(batch+1), total_acc/(batch+1)))

        scheduler.step()

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        loss_plot.append(total_loss/(batch+1))
        metric_plot.append(total_acc/(batch+1))
        print("loss : {}\t Metric : {}\t Learning_rate : {}".format(loss_plot, metric_plot, lr))

        save_model_name = save_model_path + '_' + str(epoch+1) + '.pth'
        torch.save(model.state_dict(), save_model_name)
        print("{} model saved".format(save_model_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop Disease Diagnosis')
    parser.add_argument('--mode', choices=['train', 'test'], default='test')
    parser.add_argument('--load_model_path', default='./work_dirs/efficientnet-b7_10_29_b4_c25_aug_2_4_cencrop_500_(0.9414)/_29.pth', 
                            help='Path of model weights to be loaded')
    parser.add_argument('--out_file', default='baseline.csv', 
                            help='csv file to save the result')
    parser.add_argument('--save_model_path', default='./work_dirs/efficientnet-b7/', 
                            help='Path to store model weights')
    parser.add_argument('--train_data_path', default='./data/train', 
                            help='Train Data path')
    parser.add_argument('--test_data_path', default='./data/test',
                            help='Test Data path')
    args = parser.parse_args()

    mode = args.mode
 
    load_model_path = args.load_model_path
    out_file = args.out_file
    save_model_path = args.save_model_path
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path

    if mode == 'test':
        print("Testing!!!")
    
    if mode == 'train':
        mkdir_if_not_exists(save_model_path)
        print("Training!!!")
    
    main(mode)