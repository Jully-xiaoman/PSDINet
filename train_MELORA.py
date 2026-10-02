import torch
import argparse
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict
import yaml
from models import MELORA
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params

torch.manual_seed(1203)
np.random.seed(1203)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='../config.yaml', help="Path to the config.yaml file")
parser.add_argument('--dataset', type=str, default='mimic-iii-ATC3', help="dataset name")
parser.add_argument('--eval', action='store_true', help="eval mode")
parser.add_argument(
    '--ablation',
    type=str,
    default='full',
    choices=['full', 'no_fft', 'no_time', 'no_sim', 'no_time_sim', 'no_ddi_gcn'],
    help='which ablation setting to use'
)
parser.add_argument(
    '--fft_mode',
    type=str,
    default='full',
    choices=['full', 'low', 'high', 'none'],
    help='FFT mode: full / low / high / none'
)
parser.add_argument(
    '--gpu',
    type=int,
    default=None,
    help="GPU id, e.g., 0. If not set, use config['hyperparameter']['GPU_ID']"
)

args = parser.parse_args()

# 加载配置文件
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# 获取数据集路径
def get_dataset_paths(dataset, config):
    if dataset not in config:
        raise ValueError(f"Dataset {dataset} not found in config")

    dataset_config = config[dataset]
    return dataset_config['data_path'], dataset_config['voc_path'], dataset_config['ehr_adj_path'], dataset_config[
        'ddi_adj_path']

# 获取测试权重路径
def get_resume_name(dataset, config):
    if dataset not in config['test']:
        raise ValueError(f"Test data for {dataset} not found in config")

    return config['test'][dataset]

def print_config(dataset, eval_mode):
    print("\n=== Current Configuration ===")
    print(f"Dataset: {dataset}")
    print(f"Evaluation mode: {eval_mode}")
    print("=============================")

def eval(model, data_eval, voc_size, epoch, ddi_adj_path):
    # evaluate
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0
    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for adm_idx, adm in enumerate(input):

            target_output1 = model(input[:adm_idx+1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)


        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record,ddi_adj_path)

    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))

    # case_study记录
    dill.dump(case_study, open(os.path.join(
        ".", "result_ablation", args.dataset, args.ablation, args.fft_mode
    ,'case_study.pkl'), 'wb'))
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def main():
    config = load_config(args.config)

    # get hyperparameter
    hyper_config = config['hyperparameter']
    # 1) 从 config 里读默认 GPU_ID（比如 "0"）
    cfg_gpu = str(hyper_config.get('GPU_ID', '0'))

    # 2) 如果命令行指定了 --gpu，就优先用命令行
    if args.gpu is not None:
        # 约定：--gpu -1 表示强制用 CPU
        if args.gpu < 0 or not torch.cuda.is_available():
            device = torch.device('cpu')
            device_str = 'cpu'
        else:
            device = torch.device(f'cuda:{args.gpu}')
            device_str = str(args.gpu)
    else:
        # 没有传 --gpu，就按配置文件来
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{cfg_gpu}')
            device_str = cfg_gpu
        else:
            device = torch.device('cpu')
            device_str = 'cpu'

    EPOCH = hyper_config['EPOCH']
    LR = hyper_config['LR']
    Neg_Loss = hyper_config['Neg_Loss']
    DDI_IN_MEM = hyper_config['DDI_IN_MEM']
    TARGET_DDI = hyper_config['TARGET_DDI']
    T = hyper_config['T']
    decay_weight = hyper_config['decay_weight']
    model_name = hyper_config['model_name']
    emb_dim = hyper_config['emb_dim']
    sigma =  hyper_config['sigma']

    # 1) 先根据 ablation 设置衰减和图的开关
    if args.ablation == 'full':
        use_time_decay = True
        use_sim_decay = True
        use_ddi_gcn = True

    elif args.ablation == 'no_time':
        use_time_decay = False  # 关时间衰减
        use_sim_decay = True
        use_ddi_gcn = True

    elif args.ablation == 'no_sim':
        use_time_decay = True
        use_sim_decay = False  # 关相似度衰减
        use_ddi_gcn = True

    elif args.ablation == 'no_time_sim':
        use_time_decay = False
        use_sim_decay = False  # 两个都关
        use_ddi_gcn = True

    elif args.ablation == 'no_ddi_gcn':
        use_time_decay = True
        use_sim_decay = True
        use_ddi_gcn = False  # 不用 DDI-GCN 修正，只用 EHR-GCN

    elif args.ablation == 'no_fft':
        # 这里是“完全不用 FFT”的特殊基线
        use_time_decay = True  # 你可以保持衰减模块逻辑不变
        use_sim_decay = True
        use_ddi_gcn = True
        args.fft_mode = 'none'  # ⬅ 强制关闭 FFT

    else:
        raise ValueError(f'Unknown ablation mode: {args.ablation}')

    # get dataset path
    dataset_config = config['dataset']
    data_path, voc_path, ehr_adj_path, ddi_adj_path = get_dataset_paths(args.dataset, dataset_config)

    # load dataset
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    print(type(ehr_adj))



    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    # split data for train eval and test
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    # initialize_model
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = MELORA(
        voc_size,
        ehr_adj,
        ddi_adj,
        emb_dim=emb_dim,
        sigma=sigma,
        device=device,
        ddi_in_memory=DDI_IN_MEM,
        fft_mode=args.fft_mode,
        use_time_decay=use_time_decay,
        use_sim_decay=use_sim_decay,
        use_ddi_gcn=use_ddi_gcn,
    )

    model.to(device=device)
    print('parameters', get_n_params(model))
    import sys
    sys.exit()
    optimizer = Adam(list(model.parameters()), lr=LR)

    # 记录结果的目录
    output_result_path = os.path.join(
        ".", "result_ablation", args.dataset, args.ablation, args.fft_mode
    )
    os.makedirs(output_result_path, exist_ok=True)

    # 存模型 checkpoint 的目录
    output_model_path = os.path.join(
        ".", "src", "saved_ablation", args.dataset, args.ablation, args.fft_mode
    )
    os.makedirs(output_model_path, exist_ok=True)

    if args.eval:
        print_config(args.dataset,args.eval)
        resume_name = get_resume_name(args.dataset,config)
        model.load_state_dict(torch.load(open(resume_name, 'rb')))
        eval(model, data_eval, voc_size, 0, ddi_adj_path)
    else:
        print_config(args.dataset, args.eval)
        history = {}
        best_epoch = 0
        best_ja = 0
        for epoch in range(EPOCH):
            loss_record1 = []
            start_time = time.time()
            model.train()
            prediction_loss_cnt = 0
            neg_loss_cnt = 0
            for step, input in enumerate(data_train):
                for idx, adm in enumerate(input):
                    seq_input = input[:idx+1]
                    loss1_target = np.zeros((1, voc_size[2]))
                    loss1_target[:, adm[2]] = 1
                    loss3_target = np.full((1, voc_size[2]), -1)
                    for idx, item in enumerate(adm[2]):
                        loss3_target[0][idx] = item

                    target_output1, batch_neg_loss = model(seq_input)

                    loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))
                    loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))
                    if Neg_Loss:
                        target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
                        target_output1[target_output1 >= 0.5] = 1
                        target_output1[target_output1 < 0.5] = 0
                        y_label = np.where(target_output1 == 1)[0]
                        current_ddi_rate = ddi_rate_score([[y_label]],ddi_adj_path)
                        if current_ddi_rate <= TARGET_DDI:
                            loss = 0.9 * loss1 + 0.01 * loss3
                            prediction_loss_cnt += 1
                        else:
                            rnd = np.exp((TARGET_DDI - current_ddi_rate)/T)
                            if np.random.rand(1) < rnd:
                                loss = batch_neg_loss
                                neg_loss_cnt += 1
                            else:
                                loss = 0.9 * loss1 + 0.01 * loss3
                                prediction_loss_cnt += 1
                    else:
                        loss = 0.9 * loss1 + 0.01 * loss3

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record1.append(loss.item())

                # update the printed line
                llprint('\rTrain--Epoch: %d, Step: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (epoch, step, len(data_train), prediction_loss_cnt, neg_loss_cnt))

            # annealing
            T *= decay_weight

            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch, ddi_adj_path)

            history[epoch] = {
                'ja': ja,
                'ddi_rate': ddi_rate,
                'prauc': prauc,
                'avg_p': avg_p,
                'avg_r': avg_r,
                'avg_f1': avg_f1
            }

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))
            if ja > best_ja:
                best_ja = ja
                best_epoch = epoch
                torch.save(model.state_dict(), open(os.path.join(output_result_path,'best.model'), 'wb'))

                best_metrics = {
                    'epoch': best_epoch,
                    'ja': best_ja,
                    'ddi_rate': ddi_rate,
                    'prauc': prauc,
                    'avg_p': avg_p,
                    'avg_r': avg_r,
                    'avg_f1': avg_f1
                }

                print(f"Best epoch {best_epoch} with "
                    f"Jaccard: {best_metrics['ja']:.4f}, "
                    f"DDI Rate: {best_metrics['ddi_rate']:.4f}, "
                    f"PRAUC: {best_metrics['prauc']:.4f}, "
                    f"Avg Precision: {best_metrics['avg_p']:.4f}, "
                    f"Avg Recall: {best_metrics['avg_r']:.4f}, "
                    f"Avg F1: {best_metrics['avg_f1']:.4f}")
            
            torch.save(model.state_dict(), open(os.path.join(output_model_path,'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb'))
            print('')
            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja

        # 按照组分
        dill.dump(history, open(os.path.join(output_result_path,'history_by_epoch.pkl'), 'wb'))

        print('best_epoch:', best_epoch)


if __name__ == '__main__':

    main()
