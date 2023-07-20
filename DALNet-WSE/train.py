import gc
import argparse
import collections

import pandas as pd
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler
from utils import *
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'True'
from torch.autograd import Variable
import transformers as tfs
from ranger2020 import *

import logging

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)
if __name__ == '__main__':
    # mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()

    # 模型的基本参数设备
    parser.add_argument('--seed', type=int, default=2022, help='定义随机种子')
    parser.add_argument('--vocab_size', type=int, default=30522, help='wordpiece的词汇表长度')
    parser.add_argument('--max_seq_len', type=int, default=20, help='最大的句子长度')
    parser.add_argument('--n_layers', type=int, default=2, help='交互注意力的层数')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--hidden_size', type=int, default=512, help='the size of hidden')  #512=16*32，，312=12*26
    # parser.add_argument('--head_hidden_size', type=int, default=39, help='每一个头中的隐藏层维度')
    parser.add_argument('--num_heads', type=int, default=16, help='the number of heads')
    parser.add_argument('--head_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    # parser.add_argument('--ff_size', type=int, default=312 * 4, help='FC层的过渡层的隐藏单元尺寸')
    parser.add_argument('--num_workers', type=int, default=0, help='线程数')
    parser.add_argument('--data_dir', type=str, default='VQAMED2019', help='存放数据的文件夹')
    parser.add_argument('--epochs', type=int, default=100, help='训练的周期数')
    parser.add_argument('--save_dir', type=str, default='model', help='保存模型参数的位置')
    parser.add_argument('--smoothing', type=float, default=0.1, help="label smoothing")
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--grad_num', type=int, default=4)
    parser.add_argument('--states', type=str, default='log/vqamed/yesno.log',
                        choices=['log/vqamed/plane.log',
                                 'log文件/No-C-A/Kvasir.log',
                                 'log文件/norm-C-A/Kvasir.log'])

    # parser.add_argument('--run_name', type=str, required=True, default='MVQA', help='运行的wandb的名称')

    # 模型训练的相关参数设置
    # parser.add_argument('--mixed_precision', action='store_true', default=False, help='是否使用最小准确度的问题')
    parser.add_argument('--Train', type=str, default=True, help='表示是否是进行训练')

    parser.add_argument('--category', type=str, required=False, default='yesno',
                        choices=['modality', 'plane', 'organ', 'abnormality', 'yesno'],
                        help='choose specific category if you want')
    # parser.add_argument('--patience', type=int, default=10, help='patience for rlp')
    # parser.add_argument('--factor', type=float, default=0.1, help='factor for rlp')

    args = parser.parse_args()

    # seed_everything(args.seed)

    train_df, val_df, test_df = load_data(args)

    if args.category:
        '''
        限定训练中的类别问题，如果存在类别，首先去除每一个中含有'yes'或者'no'的类，针对某一个类别做分类操作
        modality：答案类别有45个  不包含了yes/no
        plane:  答案类别有16个  在plane类中没有yes/no类型的答案
        organ:  答案类别有17个  在organ类中没有yes/no类型的答案
        abnormality：不包含yes/no中有 1/1669.1.05/1669
        '''
        train_df = train_df[train_df['category'] == args.category].reset_index(drop=True)  # 重新标注索引
        val_df = val_df[val_df['category'] == args.category].reset_index(drop=True)
        test_df = test_df[test_df['category'] == args.category].reset_index(drop=True)

    # train_answer = list(train_df['answer'])
    # train_answer = collections.Counter(train_answer)
    # answer_number = list(train_answer.values())
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(3,6))
    # plt.boxplot(answer_number,labels=['Number of categories'])
    # plt.show()
    #
    # plt.hist(answer_number, histtype='stepfilled', orientation='horizontal')
    # plt.show()

    df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    ans2idx = {ans: idx for idx, ans in enumerate(df['answer'].unique())}  # nuique()判断唯一的问题
    # test_ans2idx = {ans: idx for idx, ans in enumerate(test_df['answer'].unique())}
    idx2ans = {idx: ans for ans, idx in ans2idx.items()}  # 返回ans2idx的元素(answer: 2(数字))
    df['answer'] = df['answer'].map(ans2idx).astype(int)  # 将原本的答案转换为对应的数值

    train_df = df[df['mode'] == 'train'].reset_index(drop=True)
    val_df = df[df['mode'] == 'val'].reset_index(drop=True)
    test_df = df[df['mode'] == 'test'].reset_index(drop=True)

    args.num_classes = len(ans2idx)  # 答案类型的长度
    # args.num_test = len(test_ans2idx)
    # train = list(train_df['answer'])
    # import collections
    # import matplotlib.pyplot as plt
    #
    # number = collections.Counter(train)
    # dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)  #排序
    # number2 = collections.Counter(list(number.values()))

    # y=np.arange(max(dict.values()))
    # x = list(number2.values())
    # y = list(number2.keys())
    # xx = [104, 5]
    # yy = np.arange(1,3).astype(dtype=np.str)
    # plt.barh(yy, xx)
    # plt.show()

    # print(args.num_classes)
    # traindataset = VQAMed2019(train_df, args=args)
    # valdataset = VQAMed2019(val_df, args=args)
    testdataset = VQAMed2019(test_df, args=args)
    alldataset = VQAMed2019(df, args=args)

    # trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # valloader = DataLoader(valdataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    allloader = DataLoader(alldataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # img, sens, tokens, question_mask, img_mask, answer = next(iter(testloader))
    # print(img)

    model = DALNet_WSE(args)

    model.classifer[-1] = nn.Linear(args.hidden_size, args.num_classes)
    # model_sate_dict = torch.load('VQAMED2019/model/DAL12abnormality0.184.pt')
    # model.load_state_dict(model_sate_dict)
    model = model.to(device)

    # total_steps = len(trainloader) * args.epochs
    total_steps = len(allloader)*args.epochs
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = Ranger21(model.parameters(), lr=args.lr,
    #                                           num_epochs=args.epochs,
    #                                           num_batches_per_epoch=len(trainloader))
    # optimizer = Ranger(model.parameters(), lr=args.lr)


    scheduler = tfs.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps,num_cycles=0.5)
    scaler = GradScaler()

    if args.smoothing:
        criterion = LabelSmoothing(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # best_acc2 = 0.819
    # best_loss = np.inf
    # counter = 0


    # filename = args.states
    # logger = get_logger(filename)
    # logger.info(f'''Starting training:
    #         Epochs:          {args.epochs}
    #         Batch size:      {args.batch_size}
    #         Learning rate:   {args.lr}
    #         num_layers:      {args.n_layers}
    #         hidden_size:     {args.hidden_size}
    #         Device:          {device.type}
    #         Weight_decay:    {args.weight_decay}
    #         Grad_accumul:    {args.grad_num}
    #         augmentation:    True
    #         Dropout:         {args.dropout}
    #
    #     ''')
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')

        # train_loss, train_acc, train_bleu = train_one_epoch(trainloader, model, optimizer, criterion, scaler, scheduler, idx2ans, args)
        # val_loss, val_acc, val_bleu = validate(valloader, model, criterion, args, idx2ans)
        train_loss, train_acc, train_bleu = train_one_epoch(allloader, model, optimizer, criterion, scaler, scheduler,
                                                            idx2ans, args)
        acc, bleu, auc,wbss = test(testloader, model, args, idx2ans)

        # if not args.category:
        #     print(
        #         f'train_loss:{train_loss}, train_acc:{train_acc}, val_all_acc:{val_acc}, val_all_bleu:{val_bleu}, all_acc:{acc}, all_bleu:{bleu}')
        #
        # else:
        #     print(f'train_loss:{train_loss},val_{args.category}_acc: {val_acc:.3f},val_{args.category}_bleu: {val_bleu:.3f}')

        # print(f'acc: {acc:.3f}, bleu: {bleu:.3f}, auc: {auc:.3f}, wbss: {wbss: .3f}')
        print(f'acc: {acc:.3f}, bleu: {bleu:.3f}, wbss: {wbss: .3f}, auc: {auc: .3f}')
        # print(best_acc2)
        # if acc>0.184:
            # torch.save(model.state_dict(), os.path.join('./model/', f'{args.category}.pth'))
            # torch.save(model.state_dict(), f'VQAMED2019/model/DAL16{args.category}{acc:.3f}.pt')
            # print(f'gt: {gt}, pred: {pred}')
        # logger.info('epoch: {}, acc: {:.3f}, bleu: {:.3f}, sen: {:.3f}, spe: {:.3f}, auc: {:.3f}, wbss: {:.3f}'.format(epoch + 1, acc.item(), bleu.item(),sen.item(), spe.item(),auc.item(), wbss.item()))

        del train_loss, train_acc, train_bleu, acc, bleu, wbss, auc

        gc.collect()
        torch.cuda.empty_cache()
