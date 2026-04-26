# nesy-ai
Dual Process Thinking with Structured Moral Framework in Language Models for Moral Reasoning

Link update progress as of 17 Apr 2026:
[Link Commit](https://github.com/steven-fo/nesy-ai/commit/1233e1a97952a434f2eb6ae99ba6a03a0257882a)

Overleaf link:
[Overleaf](https://www.overleaf.com/read/xppksssvdvzc#0a402e)

Canva link:
[Canva](https://www.canva.com/design/DAHAgqwQErY/kCNFqOugpoi8zX8AcmaXQA/view?utm_content=DAHAgqwQErY&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h4c89f52007)


This repo provides code to decouple the contribution of knowledge and reasoning. The dataset used in this paper is in ./data/input. 


## Infer (./src/run_infer.sh)


Prompt LLMs to generate answers under fast thinking and slow thinking

STEP 1: Open `./src/run_infer.sh`


STEP 2: Specify LLMs names in `MODELS`


STEP 3: Assign GPU in `GPU_IDS` (two A800 GPU)


STEP 4: Run `./src/run_infer.sh`


## Citation
```
@inproceedings{
  title={Decoupling Knowledge and Reasoning in LLMs: An Exploration Using Cognitive Dual-System Theory},
  author={Mutian Yang and Jiandong Gao and Ji Wu},
  booktitle={The 40th Annual AAAI Conference on Artificial Intelligence},
  year={2025},
}
```