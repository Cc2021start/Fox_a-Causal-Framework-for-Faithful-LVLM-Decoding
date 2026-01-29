import argparse
import os
import random
import json

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from vcd_add_noise import add_diffusion_noise
from vcd_sample import evolve_vcd_sampling
import logging
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
time = datetime.now().strftime('%m-%d-%H:%M')
print(time)

MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "instructblip-13b": "eval_configs/instructblip_eval_13b.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
    "llava-1.5-13b": "eval_configs/llava-1.5_eval_13b.yaml",
}

POPE_PATH = {
    "426": "/home/kdzlys/Fox/pope/coco/test_1000_err.jsonl",
    "427": "/home/kdzlys/Fox/pope/coco/test_1000_right.jsonl",
    "coco_random": "/home/kdzlys/Fox/pope/coco/coco_pope_random.json",
    "coco_popular": "/home/kdzlys/Fox/pope/coco/coco_pope_popular.json",
    "coco_adversarial": "/home/kdzlys/Fox/pope/coco/coco_pope_adversarial.json",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
       "instructblip": "USER: <ImageHere> <question> ASSISTANT:",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}


class POPEDataSet(Dataset):
    def __init__(self, pope_path, data_path, trans):
        self.pope_path = pope_path
        self.data_path = data_path
        self.trans = trans

        # 获取data_path中所有存在的图片文件名
        available_images = set()
        if os.path.exists(data_path):
            for file in os.listdir(data_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    available_images.add(file)

        image_list, query_list, label_list = [], [], []
        total_count = 0
        filtered_count = 0

        for q in open(pope_path, 'r'):
            total_count += 1
            line = json.loads(q)


            if line['image'] in available_images:
                image_list.append(line['image'])
                query_list.append(line['text'])
                label_list.append(line['label'])
                filtered_count += 1


        for i in range(len(label_list)):
            if label_list[i] == 'no':
                label_list[i] = 0
            else:
                label_list[i] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

        yes_count = sum(1 for label in label_list if label == 1)
        no_count = len(label_list) - yes_count

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")
        image = self.trans(raw_image)
        query = self.query_list[index]
        label = self.label_list[index]

        return {"image": image, "query": query, "label": label}


def parse_args():
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model", type=str, default="llava-1.5", help="model")
    parser.add_argument("--pope-type", type=str, default="coco_random", help="model")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--data-path", type=str, default="/data-disk/cc/val2014", help="data path")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="num workers")
    parser.add_argument("--answers-file", type=str, default="/data-disk/results")
    # vision contrastive decoding
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use-cd", action='store_true', default=True)
    parser.add_argument("--use-icd", action='store_true', default=False)
    parser.add_argument("--use-vcd", action='store_true', default=False)
    parser.add_argument("--sample-greedy", action='store_true', default=False)
    # fast token merging
    parser.add_argument("--use-fast-v", action='store_true', default=False)
    parser.add_argument("--fast-v-inplace", default=False)
    parser.add_argument("--fast-v-attention-rank", type=int, default=100)
    parser.add_argument("--fast-v-attention-rank-add", type=int, default=100)
    parser.add_argument("--fast-v-agg-layer", type=int, default=2)
    # auto-generation
    parser.add_argument("--fast-v-sys-length", default=None, type=int, help='the length of system prompt')
    parser.add_argument("--fast-v-image-token-length", default=None, type=int, help='the length of image token')
    # opera-beamsearch
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--sample", action='store_true', default=True)
    parser.add_argument("--scale-factor", type=float, default=50)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--num_attn-candidates", type=int, default=5)
    parser.add_argument("--penalty-weights", type=float, default=1)
    parser.add_argument("--opera", action='store_true', default=False)

    parser.add_argument("--cd-alpha", type=float, default=1)
    parser.add_argument("--cd-beta", type=float, default=0.1)
    parser.add_argument("--options")
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def print_acc(pred_list, label_list, logger, error_cases=None):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP) if (TP + FP) > 0 else 0
    recall = float(TP) / float(TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    acc = (TP + TN) / (TP + TN + FP + FN)

    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('Total errors: {}'.format(FP + FN))
    print('False Positives: {}'.format(FP))
    print('False Negatives: {}'.format(FN))

    logger.info('Accuracy: {}'.format(acc))
    logger.info('Precision: {}'.format(precision))
    logger.info('Recall: {}'.format(recall))
    logger.info('F1 score: {}'.format(f1))
    logger.info('Yes ratio: {}'.format(yes_ratio))
    logger.info('Total errors: {}'.format(FP + FN))
    logger.info('False Positives: {}'.format(FP))
    logger.info('False Negatives: {}'.format(FN))

    # 如果提供了错误案例，打印前10个
    if error_cases:
        print("\n=== Error Cases Examples (first 10) ===")
        for i, case in enumerate(error_cases[:10]):
            print(f"Error {i + 1}: Image: {case.get('image_name', 'unknown')}")
            print(f"  Question: {case.get('question', 'unknown')}")
            print(f"  Answer: {case.get('answer', 'unknown')}")
            print(
                f"  Predicted: {case.get('predicted', 'unknown')}, Ground Truth: {case.get('ground_truth', 'unknown')}")
            print(f"  Error Type: {case.get('error_type', 'unknown')}")
            print("-" * 50)


def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out:
        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')
        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
    return pred_list


def main():
    # 设置参数
    args = parse_args()

    def log_string(str):
        logger.info(str)
        print(str)
    exp_dir = Path(os.path.join('results', 'log'))
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir = exp_dir.joinpath(args.pope_type)
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger(args.model)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
    args.pope_path = POPE_PATH[args.pope_type]
    cfg = Config(args)
    cfg.config.model.vit_model = "/home/kdzlys/LLava_1.5_model/clip-vit-large-patch14-336"
    cfg.config.preprocess.vis_processor.train.proc_type = "/home/kdzlys/LLava_1.5_model/clip-vit-large-patch14-336"
    cfg.config.preprocess.vis_processor.eval.proc_type = "/home/kdzlys/LLava_1.5_model/clip-vit-large-patch14-336"
    setup_seeds(cfg)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    if args.sample or (args.use_icd or args.use_cd or args.use_vcd or args.sample_greedy) == True:
        evolve_vcd_sampling()

    # ========================================
    #             Model Initialization
    # ========================================
    print('Initializing Model')

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    # set model decoding config
    if args.model == "instructblip":
        if args.use_fast_v == True:
            model.llm_model.config.use_fast_v = args.use_fast_v
            model.llm_model.config.fast_v_inplace = args.fast_v_inplace
            model.llm_model.config.fast_v_sys_length = args.fast_v_sys_length
            model.llm_model.config.fast_v_image_token_length = args.fast_v_image_token_length
            model.llm_model.config.fast_v_attention_rank = args.fast_v_attention_rank
            model.llm_model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
            model.llm_model.config.fast_v_agg_layer = args.fast_v_agg_layer
            model.llama_model.config.fast_v_agg_layer = args.fast_v_agg_layer
        else:
            model.llm_model.config.use_fast_v = args.use_fast_v
        model.llm_model.model.reset_fastv()
    else:
        if args.use_fast_v == True:
            model.llama_model.config.use_fast_v = args.use_fast_v
            model.llama_model.config.fast_v_inplace = args.fast_v_inplace
            model.llama_model.config.fast_v_sys_length = args.fast_v_sys_length
            model.llama_model.config.fast_v_image_token_length = args.fast_v_image_token_length
            model.llama_model.config.fast_v_attention_rank = args.fast_v_attention_rank
            model.llama_model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
            model.llama_model.config.fast_v_agg_layer = args.fast_v_agg_layer
        else:
            model.llama_model.config.use_fast_v = args.use_fast_v
            model.llama_model.config.fast_v_agg_layer = args.fast_v_agg_layer
        model.llama_model.model.reset_fastv()

    vis_processors, txt_processors = load_preprocess(cfg.get_config().preprocess)
    print(vis_processors["eval"].transform)
    print("Done!")

    # load pope data (现在会自动筛选只包含500张图片的数据)
    pope_dataset = POPEDataSet(
        pope_path=args.pope_path,
        data_path=args.data_path,
        trans=vis_processors["eval"]
    )
    pope_loader = torch.utils.data.DataLoader(        pope_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    print("load data finished")

    # 设置保存路径
    os.makedirs("/home/kdzlys/Fox/log/llava_pope", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_save_path = f"/home/kdzlys/Fox/only/POPE/LLAVA/消融_our——0.1_{args.pope_type}.jsonl"

    print("Start eval...")
    pred_list, pred_list_s, label_list = [], [], []
    all_results = []
    error_cases = []
    sample_idx = 0
    for layer in model.llama_model.model.layers:
        layer.self_attn.use_phj = True

    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        image = data["image"]
        qu = data["query"]
        label = data["label"]
        label_list = label_list + list(label)

        # 获取图像文件名
        image_names = []
        for i in range(len(qu)):
            current_sample_idx = sample_idx + i
            if current_sample_idx < len(pope_dataset.image_list):
                image_names.append(pope_dataset.image_list[current_sample_idx])
            else:
                image_names.append(f"batch_{batch_id}_{i}")

        template = INSTRUCTION_TEMPLATE[args.model]
        qu = [template.replace("<question>", q) for q in qu]

        image = image.to(device)
        label = torch.Tensor(label).to(device)

        if args.use_icd:
            text_cd = 'You are a confused object detector.'
            if args.model == 'shikra':
                prompt_cd = qu[0].split("<im_end>")[0] + "<im_end>" + ' ' + text_cd + qu[0].split("<im_end>")[-1]
            elif args.model == 'llava-1.5' or args.model == 'instructblip':
                prompt_cd = qu[0].split("<ImageHere>")[0] + "<ImageHere>" + ' ' + text_cd + qu[0].split("<ImageHere>")[
                    -1]
            # elif args.model == 'lrv_instruct' or args.model == 'minigpt4':
            else:
                prompt_cd = qu[0].split("</Img>")[0] + "</Img>" + ' ' + text_cd + qu[0].split("</Img>")[-1]
        else:
            text_cd = None

        if args.use_cd:
            image_cd = image.to(device)
        elif args.use_vcd:
            image_cd = add_diffusion_noise(image, args.noise_step)
            image_cd = image_cd.to(device)
        else:
            image_cd = None

        with torch.enable_grad():
            out = model.generate(
                prompt=qu,
                image=image.half(),
                images_cd=(image_cd.half() if image_cd is not None else None),
                prompt_cd=(prompt_cd if text_cd is not None else None),
                use_nucleus_sampling=args.sample,
                num_beams=args.beam,  # 这里使用束搜索，候选路径数由args.beam决定
                max_new_tokens=1,
                output_attentions=True,
                opera_decoding=args.opera,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
                use_cache=False,
                sample_greedy=args.sample_greedy
            )

        preds = []
        preds = recorder(out, preds)

        # 即时写入 JSON 文件并收集错误案例
        with open(json_save_path, "a", encoding="utf-8") as f:
            for i in range(len(out)):
                current_image_name = image_names[i] if i < len(image_names) else f"unknown_{batch_id}_{i}"
                current_label = int(label[i].item())
                current_pred = preds[i]
                print(current_pred)

                # 保存完整结果
                result = {
                    "id": sample_idx + i,
                    "image_name": current_image_name,
                    "query": qu[i],
                    "model_output": out[i],
                    "prediction": current_pred,
                    "label": current_label
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                all_results.append(result)


        pred_list.extend(preds)
        sample_idx += len(qu)

    print("[{}, {}]===============================================".format(args.scale_factor, args.num_attn_candidates))


    print(f'All results saved to: {json_save_path}')
    logger.info(f'All results saved to: {json_save_path}')

    if len(pred_list) != 0:
        print_acc(pred_list, label_list, logger, error_cases)

    if len(pred_list_s) != 0:
        print_acc(pred_list_s, label_list, logger, error_cases)


if __name__ == "__main__":
    main()