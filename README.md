Fox: A Causal Framework for Faithful LVLM Decoding
Official implementation of the paper: Dismantling Pathological Shortcuts: A Causal Framework for Faithful LVLM Decoding
🔍 Overview
<img width="898" height="264" alt="image" src="https://github.com/user-attachments/assets/3b954bc3-b188-4f35-bb8e-01eaf4991b7d" />
We model LVLM token generation via structural causal graphs, revealing that risky attention heads act as pathological shortcut mediators to induce hallucinations, and mitigate this issue through soft causal suppression intervention on such high‑risk mediators.

<img width="1282" height="399" alt="image" src="https://github.com/user-attachments/assets/cc7a8ecc-e233-4f86-a664-fdace02cc92d" />

This pipeline illustrates our inference‑time faithful decoding paradigm, which sequentially conducts high‑risk mediator diagnosis, head‑level causal intervention, and conflict‑gated distribution fusion to suppress LVLM hallucinations while preserving generation fluency.

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
