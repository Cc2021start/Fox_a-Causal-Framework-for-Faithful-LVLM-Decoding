Fox: A Causal Framework for Faithful LVLM Decoding
Official implementation of the paper: Dismantling Pathological Shortcuts: A Causal Framework for Faithful LVLM Decoding
🔍 Overview
Large Vision-Language Models (LVLMs) often suffer from severe object hallucination, generating text inconsistent with visual inputs. Recent inference-time mitigation methods (e.g., VCD, SID) have made great progress by perturbing hidden representations or suppressing biased tokens.
However, we reveal that existing methods rely on the attention intensity assumption, which overlooks the structural shortcut problem: risky attention heads directly capture language priors instead of visual evidence, dominating token generation.
To address this issue, we propose Fox, a training-free causal framework for faithful LVLM decoding.
Fox diagnoses risky attention heads that form pathological shortcuts, performs targeted causal intervention to block prior-driven hallucination paths, and fuses observational and interventional distributions for balanced faithfulness and generation fluency.
🧠 Core Methodology & Visualization
Structural Causal Model & Intervention Paradigm
(Figure 1: Left: Observation phase, risky mediator heads 
H 
R
​
 
 take the shortcut path to dominate generation. Right: Our intervention paradigm, soft causal suppression on risky mediators blocks pathological shortcuts.)
Overall Pipeline of Fox
(Figure 2: End-to-end pipeline of Fox, including high-risk mediator diagnosis, head-level causal intervention, and conflict-gated cooperative decoding.)
✨ Key Contributions
We challenge the mainstream attention intensity assumption in LVLM hallucination research, and identify the structural shortcut issue caused by risky attention heads.
We propose a causal diagnosis strategy to unsupervisedly locate high-risk attention heads that dominate hallucinatory generation.
We design a soft causal suppression intervention and conflict-gated cooperative decoding, achieving state-of-the-art performance without extra training.
We extensively validate Fox on POPE, CHAIR, MME benchmarks, with strong improvements over previous inference-time baselines including VCD and SID.
🙏 Acknowledgements
Our work is built upon and benefits greatly from previous outstanding open-source projects:
VCD: Visual Contrastive Decoding for mitigating hallucinations in LVLMs
SID: Stable Intervention Decoding for faithful generation of vision-language models
We sincerely thank the authors of VCD and SID for their valuable research and open-source code, which inspire our causal intervention design for LVLM decoding.
🚀 Quick Start
1. Clone Repository
bash
运行
git clone https://github.com/Cc2021start/Fox_a-Causal-Framework-for-Faithful-LVLM-Decoding.git
cd Fox_a-Causal-Framework-for-Faithful-LVLM-Decoding
2. Environment Setup
bash
运行
conda env create -f environment.yml
conda activate fox
3. Run Evaluation & Generation
bash
运行
# Evaluate on POPE benchmark
python pope_eval.py

# Evaluate captioning performance on CHAIR
python chair_eval.py

# Generate samples with Fox decoding
python vcd_sample.py
python vcd_add_noise.py
📁 Project Structure
plaintext
Fox/
├── minigpt4/           # Backbone LVLM model code
├── eval_configs/       # Evaluation configuration files
├── pope/               # POPE benchmark dataset
├── transformers/       # Modified transformer with causal intervention
├── pope_eval.py        # POPE hallucination evaluation
├── chair_eval.py       # CHAIR captioning evaluation
├── vcd_sample.py       # Sampling pipeline with Fox
├── vcd_add_noise.py    # Hidden representation intervention
├── environment.yml     # Conda environment file
└── README.md           # Project documentation
📊 Experimental Results
We compare Fox with mainstream inference-time baselines (VCD, SID, vanilla LLaVA-1.5) on multiple benchmarks:
POPE (Adversarial Set): Outperforms VCD & SID with higher accuracy and lower hallucination ratio
CHAIR: Achieves significant relative improvement on hallucination metrics
MME: Consistent gains on existence, color, position reasoning tasks
Efficiency: Near real-time inference speed, compatible with practical deployment
📝 Citation
If you find our work useful for your research, please cite our paper:
bibtex
@inproceedings{fox2026causal,
  title={Dismantling Pathological Shortcuts: A Causal Framework for Faithful LVLM Decoding},
  author={Your Authors},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
