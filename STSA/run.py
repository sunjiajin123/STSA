import os
import gc
import time
import random
import logging
import torch
import pynvml
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config_tune import ConfigTune
from config.config_regression import ConfigRegression
from config.config_classification import ConfigClassification

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def setup_seed(seed):                  #设置种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args):

    # 模型存放路径
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir,f'{args.modelName}-{args.datasetName}-{args.train_mode}.pth')




    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device


    # 加载数据与模型
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)


    # 日志输出模型参数量
    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
                # print(p)
        return answer
    logger.info(f'The model has {count_parameters(model)} trainable parameters')

    # 加载训练框架
    atio = ATIO().getTrain(args)

    # 开始训练
    atio.do_train(model, dataloader)                                                                                    #开始训练
    # load pretrained model
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))                                                             #读取模型
    model.to(device)

    #此处可加入可视化


    if args.is_tune:
        # using valid dataset to tune hyper parameters
        results = atio.do_test(model, dataloader['valid'], mode="VALID")
    else:

        # 模型+测试集     获得测试集结果
        results = atio.do_test(model, dataloader['test'], mode="TEST")


    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)
 
    return results                                                                                                      #返回测试集结果

def run_tune(args, tune_times=50):
    args.res_save_dir = os.path.join(args.res_save_dir, 'tunes')
    init_args = args
    has_debuged = [] # save used paras
    save_file_path = os.path.join(args.res_save_dir, \
                                f'{args.datasetName}-{args.modelName}-{args.train_mode}-tune.csv')
    if not os.path.exists(os.path.dirname(save_file_path)):
        os.makedirs(os.path.dirname(save_file_path))
    
    for i in range(tune_times):
        # cancel random seed
        setup_seed(int(time.time()))
        args = init_args
        config = ConfigTune(args)
        args = config.get_config()
        # print debugging params
        logger.info("#"*40 + '%s-(%d/%d)' %(args.modelName, i+1, tune_times) + '#'*40)
        for k,v in args.items():
            if k in args.d_paras:
                logger.info(k + ':' + str(v))
        logger.info("#"*90)
        logger.info('Start running %s...' %(args.modelName))
        # restore existed paras
        if i == 0 and os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
            for i in range(len(df)):
                has_debuged.append([df.loc[i,k] for k in args.d_paras])
        # check paras
        cur_paras = [args[v] for v in args.d_paras]
        if cur_paras in has_debuged:
            logger.info('These paras have been used!')
            time.sleep(3)
            continue
        has_debuged.append(cur_paras)
        results = []
        for j, seed in enumerate([1111]):
            args.cur_time = j + 1
            setup_seed(seed)
            results.append(run(args))
        # save results to csv
        logger.info('Start saving results...')
        if os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
        else:
            df = pd.DataFrame(columns = [k for k in args.d_paras] + [k for k in results[0].keys()])
        # stat results
        tmp = [args[c] for c in args.d_paras]
        for col in results[0].keys():
            values = [r[col] for r in results]
            tmp.append(round(sum(values) * 100 / len(values), 2))

        df.loc[len(df)] = tmp
        df.to_csv(save_file_path, index=None)
        logger.info('Results are saved to %s...' %(save_file_path))

def run_normal(args):

    # 将结果存放到results/20200506/normals文件夹
    args.res_save_dir = os.path.join(args.res_save_dir, 'normals')
    init_args = args
    model_results = []
    seeds = args.seeds

    # 根据不同的随机种子重复5次
    for i, seed in enumerate(seeds):
        args = init_args

        # 根据回归还是分类，设置参数,ConfigRegression会返回我们所选模型的训练参数
        if args.train_mode == "regression":
            config = ConfigRegression(args)
        else:
            config = ConfigClassification(args)
        args = config.get_config()

        # 设置随机种子
        setup_seed(seed)
        args.seed = seed

        # 日志 开始+参数信息
        logger.info('Start running %s...' %(args.modelName))
        logger.info(args)

        # 转入到run()
        args.cur_time = i+1
        test_results = run(args)                                                                             #进入到run()函数，并返回结果
        # 保存结果
        model_results.append(test_results)

    criterions = list(model_results[0].keys())
    # load other results
    save_path = os.path.join(args.res_save_dir, \
                        f'{args.datasetName}-{args.train_mode}.csv')                                         #保存路径
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    # save results
    res = [args.modelName]
    for c in criterions:                                                                                    #将5次随机数结果求平均值，保存到results中
        values = [r[c] for r in model_results]
        mean = round(np.mean(values)*100, 2)
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Results are added to %s...' %(save_path))

def set_log(args):                                                                                              #日志
    log_file_path = f'logs/{args.modelName}-{args.datasetName}.log'
    # set logging
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    # formatter_stream = logging.Formatter('%(message)s')
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    # ch.setFormatter(formatter_stream)
    # logger.addHandler(ch)
    return logger

def worker(cur_task=None):

    # 读取预参数
    args = parse_args()
    global logger
    if cur_task:
        stime = random.random()*60
        print(f"{os.getpid()} process will wait: {stime} seconds...")
        time.sleep(stime) # avoid use the same gpu at first
        args.is_tune = True if cur_task['is_tune'] else False
        args.train_mode = cur_task['train_mode']
        args.modelName = cur_task['modelName']
        args.datasetName = cur_task['datasetName']
        try:
            logger = set_log(args)
            args.seeds = [1112, 1113, 1114, 1115,1111]
            if args.is_tune:
                run_tune(args, tune_times=cur_task['tune_times'])
            else:
                run_normal(args)
            df = pd.read_csv('tasks.csv')
            df.loc[cur_task['index'], 'state'] = 1  # 任务完成
        except Exception as e:
            logger.error(e)
            df = pd.read_csv('tasks.csv')
            df.loc[cur_task['index'], 'state'] = -1 # 任务出错
            df.loc[cur_task['index'], 'error_info'] = str(e)
        finally:
            df.to_csv('tasks.csv', index=None)
    else:
        logger = set_log(args)                                                                                      #创建日志文件(模型名，数据集，路径)
        args.seeds = [1111,1112, 1113, 1114, 1115]                                                                  #设置随机种子
        if args.is_tune:
            # run_tune(args, tune_times=cur_task['tune_times'])
            run_tune(args, tune_times=50)
        else:
            run_normal(args)

def parse_args():                                     #预参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--need_task_scheduling', type=bool, default=False,
                        help='use the task scheduling module.')                                                     #使用任务调度模块
    parser.add_argument('--is_tune', type=bool, default=False,
                        help='tune parameters ?')                                                                    #微调
    parser.add_argument('--train_mode', type=str, default="classification",
                        help='regression / classification')                                                          #回归/分类
    parser.add_argument('--modelName', type=str, default='stsa_s',
                        help='support lf_dnn/ef_lstm/tfn/lmf/mfn/graph_mfn/mult/misa/mlf_dnn/mtfn/mlmf/self_mm')     #选择模型
    parser.add_argument('--datasetName', type=str, default='mosei',
                        help='support mosi/mosei/sims')                                                              #选择数据集
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')                                                                #训练完模型后存放的路径
    parser.add_argument('--res_save_dir', type=str, default='results/20200506',
                        help='path to save results.')                                                                #存放结果
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')            #选择gpu
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.need_task_scheduling:                     #是否按照task.csv计划表的顺序训练
        mp.set_start_method('spawn')
        # load uncompleted tasks
        df = pd.read_csv('tasks.csv')
        left_tasks = []
        for index in range(len(df)):
            if df.loc[index, 'state'] == 0:
                cur_task = {name:df.loc[index,name] for name in df.columns}
                cur_task['index'] = index
                left_tasks.append(cur_task)
        # create process pools
        po = Pool(4)
        for cur_task in left_tasks:
            po.apply_async(worker, (cur_task,))
        # close and wait
        print('-----start--------')
        po.close()
        po.join()
        print('-----end--------')
    else:
        # 开始训练
        worker()


#整体流程：1.parse_args() 第一次设置参数

#        main->woker->runnormal

#        在woker中,创建日志文件,并设置好随机数

#        在runnormal中,根据随机种子循环5次，在每次循环中,(先根据参数判断是回归还是分类，然后获取对应的具体参数，然后设置随机种子，记录日志，通过run方法返回test集的结果，加入到model_result列表中)
#        循环结束后，5次随机数结果求平均值，保存到results中

#        在run中，先设置模型存放路径，再配置gpu，根据参数获得数据加载器(load_data.py)，根据参数获得AMIO模型,日志输出模型参数量，创建模型训练对象，开始训练(将model和data放入ATIO中),将训练完的模型，去跑测试集，返回测试集结果

#        具体训练流程：参考trains/TFN的注释