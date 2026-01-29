import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from datetime import datetime
import json
import warnings
from vcd_add_noise import add_diffusion_noise
from vcd_sample import evolve_vcd_sampling

# Initialize VCD sampling modification
evolve_vcd_sampling()
current_time = datetime.now().strftime('%m-%d-%H:%M')
print(current_time)

warnings.filterwarnings("ignore")

MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "USER: <ImageHere> <question> ASSISTANT:",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, default="llava-1.5", help="model")
parser.add_argument("--gpu-id", type=int, default=1, help="specify the gpu to load the model.")
parser.add_argument("--noise-step", type=int, default=500)
parser.add_argument("--use-cd", action='store_true', default=False)
parser.add_argument("--data-path", type=str, default="/data-disk/only_test", help="data path")
parser.add_argument("--use-icd", action='store_true', default=False)
parser.add_argument("--use-vcd", action='store_true', default=False)
parser.add_argument("--cd-alpha", type=float, default=1)
parser.add_argument("--cd-beta", type=float, default=0.1)
parser.add_argument("--sample-greedy", default=True)
parser.add_argument("--options", nargs="+", help="override settings in config")
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--num-workers", type=int, default=1)
parser.add_argument("--use-fast-v", action='store_true', default=False)
parser.add_argument("--fast-v-inplace", default=False)
parser.add_argument("--fast-v-attention-rank", type=int, default=57)
parser.add_argument("--fast-v-attention-rank-add", type=int, default=57)
parser.add_argument("--fast-v-agg-layer", type=int, default=2)
parser.add_argument('--fast-v-sys-length', default=None, type=int)
parser.add_argument('--fast-v-image-token-length', default=None, type=int)
parser.add_argument("--test-sample", type=int, default=500)
parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--sample", type=bool, default=True)
parser.add_argument("--scale-factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num-attn-candidates", type=int, default=5)
parser.add_argument("--penalty-weights", type=float, default=1.0)
parser.add_argument("--opera", action='store_true', default=False)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.models import load_preprocess

cfg = Config(args)
# Paths to local CLIP weights
cfg.config.model.vit_model = "/home/kdzlys/LLava_1.5_model/clip-vit-large-patch14-336"
cfg.config.preprocess.vis_processor.train.proc_type = "/home/kdzlys/LLava_1.5_model/clip-vit-large-patch14-336"
cfg.config.preprocess.vis_processor.eval.proc_type = "/home/kdzlys/LLava_1.5_model/clip-vit-large-patch14-336"

setup_seeds(cfg)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

print('Initializing Model')
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()

# Configure decoding settings for FastV
if args.model == "instructblip":
    model.llm_model.config.use_fast_v = args.use_fast_v
    if args.use_fast_v:
        model.llm_model.config.fast_v_inplace = args.fast_v_inplace
        model.llm_model.config.fast_v_sys_length = args.fast_v_sys_length
        model.llm_model.config.fast_v_image_token_length = args.fast_v_image_token_length
        model.llm_model.config.fast_v_attention_rank = args.fast_v_attention_rank
        model.llm_model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
        model.llm_model.config.fast_v_agg_layer = args.fast_v_agg_layer
    model.llm_model.model.reset_fastv()
else:
    model.llama_model.config.use_fast_v = args.use_fast_v
    if args.use_fast_v:
        model.llama_model.config.fast_v_inplace = args.fast_v_inplace
        model.llama_model.config.fast_v_sys_length = args.fast_v_sys_length
        model.llama_model.config.fast_v_image_token_length = args.fast_v_image_token_length
        model.llama_model.config.fast_v_attention_rank = args.fast_v_attention_rank
        model.llama_model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
        model.llama_model.config.fast_v_agg_layer = args.fast_v_agg_layer
    else:
        model.llama_model.config.fast_v_agg_layer = args.fast_v_agg_layer
    model.llama_model.model.reset_fastv()

processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)

img_files = os.listdir(args.data_path)
random.shuffle(img_files)

base_dir = "/home/kdzlys/Fox/final_results/GPT4/llava"
os.makedirs(base_dir, exist_ok=True)

for img_idx in tqdm(range(len(img_files))):
    torch.cuda.empty_cache()
    if img_idx == args.test_sample:
        break

    img_file = img_files[img_idx]
    img_id_num = int(img_file.split(".jpg")[0][-6:])

    img_save = {"image_id": img_id_num}

    image_path = os.path.join(args.data_path, img_file)
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # Question setup
    qu = "Please describe this image in detail."
    template = INSTRUCTION_TEMPLATE[args.model]
    prompt = [template.replace("<question>", qu)]

    # Contrastive Decoding setup (ICD)
    prompt_cd = None
    if args.use_icd:
        text_cd = 'You are a confused object detector.'
        if args.model == 'shikra':
            prompt_cd = qu[0].split("<im_end>")[0] + "<im_end>" + ' ' + text_cd + qu[0].split("<im_end>")[-1]
        elif args.model == 'llava-1.5' or args.model == 'instructblip':
            prompt_cd = qu[0].split("<ImageHere>")[0] + "<ImageHere>" + ' ' + text_cd + qu[0].split("<ImageHere>")[-1]
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

    with torch.inference_mode():
        out = model.generate(
            prompt=prompt,
            image=norm(image).half(),
            images_cd=(image_cd.half() if image_cd is not None else None),
            prompt_cd=prompt_cd,
            use_nucleus_sampling=args.sample,
            num_beams=args.beam,
            max_new_tokens=512,
            output_attentions=True,
            opera_decoding=args.opera,
            scale_factor=args.scale_factor,
            threshold=args.threshold,
            num_attn_candidates=args.num_attn_candidates,
            penalty_weights=args.penalty_weights,
            use_cache=True,
            sample_greedy=args.sample_greedy
        )

    img_save["caption"] = out[0]
    print(out[0])

    output_file_path = os.path.join(base_dir, 'test.jsonl')
    with open(output_file_path, "a") as f:
        json.dump(img_save, f)
        f.write('\n')