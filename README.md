# LocalLLaMA tracker
Tracking Reddit sub r/LocalLLaMA Wiki coverage of AI Module, Tracker on sub for Module Release on sub (Manual Update)
Base on this wiki thread https://www.reddit.com/r/LocalLLaMA/wiki/models/ by u/Civil_Collection7267
Last Update 2 June 2023

# Disclaimer
Use at your own risk, i not endorse or support the list of module show here, use it with responsible, just like knife it double edge sword.

## Specification 

8 Bit Specification for LLMA
| Model     | VRAM Used | Minimum Total VRAM | Card examples         | RAM/Swap to Load\* |
| --------- | --------- | ------------------ | --------------------- | ------------------ |
| LLaMA-7B  | 9.2GB     | 10GB               | 3060 12GB, 3080 10GB  | 24 GB              |
| LLaMA-13B | 16.3GB    | 20GB               | 3090, 3090 Ti, 4090   | 32 GB              |
| LLaMA-30B | 36GB      | 40GB               | A6000 48GB, A100 40GB | 64 GB              |
| LLaMA-65B | 74GB      | 80GB               | A100 80GB             | 128 GB             |

* System RAM, not VRAM, required to load the model, in addition to having enough VRAM.  Not required to run the model. You can use swap space if you do not have enough RAM.   

4 Bit Specification for LLMA

| Model     | Minimum Total VRAM | Card examples                                             | RAM/Swap to Load\* |
| --------- | ------------------ | --------------------------------------------------------- | ------------------ |
| LLaMA-7B  | 6GB                | GTX 1660, 2060, AMD 5700 XT, RTX 3050, 3060               | 6 GB               |
| LLaMA-13B | 10GB               | AMD 6900 XT, RTX 2060 12GB, 3060 12GB, 3080, A2000        | 12 GB              |
| LLaMA-30B | 20GB               | RTX 3080 20GB, A4500, A5000, 3090, 4090, 6000, Tesla V100 | 32 GB              |
| LLaMA-65B | 40GB               | A100 40GB, 2x3090, 2x4090, A40, RTX A6000, 8000           | 64 GB              |

* System RAM, not VRAM, required to load the model, in addition to having enough VRAM. 
 Not required to run the model. You can use swap space if you do not have enough RAM.   


## Current Best Module

Current LLMA base on 
====================

Best choice means for most tasks. There are other options for different niches. For a model like Vicuna but with less restrictions, use GPT4 x Vicuna. For RP chatting, use base LLaMA 30B or 65B without LoRA and with a character card.

For writing stories, use the current best choice below if you want the least amount of effort for decent results. If you want highly detailed and personalized stories and don't mind spending a lot of time on prompting, use base LLaMA 30B or 65B without LoRA.

Hugging Face
------------

7B: [Vicuna 7B v1.1](https://huggingface.co/eachadea/vicuna-7b-1.1)

13B: [Vicuna 13B v1.1](https://huggingface.co/eachadea/vicuna-13b-1.1)

30B: [Guanaco](https://huggingface.co/timdettmers/guanaco-33b-merged)

65B: [Guanaco 65B](https://huggingface.co/timdettmers/guanaco-65b-merged)

7B 4-bit GPTQ: [Vicuna 7B v1.1 4-bit](https://huggingface.co/localmodels/vicuna-7b-v1.1-4bit/tree/main/gptq)

13B 4-bit GPTQ: [Vicuna 13B v1.1 4-bit](https://huggingface.co/localmodels/vicuna-13b-v1.1-4bit/tree/main/gptq)

30B 4-bit GPTQ: [GPT4 Alpaca LoRA 30B Merge](https://huggingface.co/MetaIX/GPT4-X-Alpaca-30B-4bit)\*

65B 4-bit GPTQ: [Guanaco 65B 4-bit](https://huggingface.co/localmodels/guanaco-65b-gptq)

llama.cpp
---------

7B: [Vicuna v1.1](https://huggingface.co/eachadea/ggml-vicuna-7b-1.1)

13B: [Vicuna v1.1](https://huggingface.co/eachadea/ggml-vicuna-13b-1.1)

30B: [GPT4 Alpaca LoRA Merge](https://huggingface.co/MetaIX/GPT4-X-Alpaca-30B-4bit)\*

65B: [Guanaco 65B](https://huggingface.co/localmodels/guanaco-65b-ggml)

\*Use OASST LLaMA 30B below for the closest ChatGPT clone.

* * *

All Downloads
=============

[r/LocalLLaMA](/r/LocalLLaMA) does not endorse, claim responsibility for, or associate with any models, groups, or individuals listed here. If you would like your link added or removed from this list, please send a message to modmail.

This list is not comprehensive but should include most relevant links. If you plan on copying this list to use elsewhere but won't be updating it yourself, feel free to link back to this wiki page as this will be kept updated with the latest downloads.

Some links may have multiple formats. Always use .safetensors when available.

Models
------

Base 7B-65B 4-bit without groupsize can be [downloaded here](https://github.com/oobabooga/text-generation-webui/files/11069779/LLaMA-HF-4bit.zip).

Base 7B-65B 4-bit with groupsize can be [downloaded here](https://github.com/oobabooga/text-generation-webui/files/11070361/LLaMA-HF-4bit-128g.zip).

Due to the increasing amount of models available, parts of this section have been split into charts for easier comparison. The models listed directly below have been tested for their quality.

Models listed in the Extra section are not worse than the models in the chart but are generally unique in some way. For example, MedAlpaca was made for medical domain tasks, LLaVA for visual instruction, etc.

**Sorted approximately from best to worst**, subjective comparison by category:

**7B**

**Models** restricted

**Models** unrestricted

[Vicuna 7B v1.1](https://huggingface.co/eachadea/vicuna-7b-1.1)

[WizardLM Uncensored](https://huggingface.co/ehartford/WizardLM-7B-Uncensored)2\*

[Baize V2 7B](https://huggingface.co/project-baize/baize-v2-7b)1\*

[AlpacaGPT4 7B](https://huggingface.co/LLMs/AlpacaGPT4-7B-elina)3\*

[Vicuna Evol-Instruct](https://huggingface.co/LLMs/Vicuna-EvolInstruct-7B)

[Alpaca Native](https://huggingface.co/chavinlo/alpaca-native)

[Vicuna 7B v1.1 4-bit](https://huggingface.co/localmodels/vicuna-7b-v1.1-4bit/tree/main/gptq)

[WizardLM Uncensored 4-bit](https://huggingface.co/localmodels/WizardLM-7B-Uncensored-4bit/tree/main/gptq)

[Baize V2 7B 4-bit](https://huggingface.co/localmodels/baize-v2-7b-4bit/tree/main/gptq)

[Alpaca Native 4-bit](https://huggingface.co/ozcur/alpaca-native-4bit)

[LLaMA](https://huggingface.co/elinas/llama-7b-hf-transformers-4.29)

Extra: [WizardVicunaLM Uncensored](https://huggingface.co/ehartford/Wizard-Vicuna-7B-Uncensored), [LLaMA Deus V3 Merge](https://huggingface.co/teknium/llama-deus-7b-v3-lora-merged), [Pygmalion 7B](https://huggingface.co/Neko-Institute-of-Science/pygmalion-7b), [Pygmalion 7B 4-bit](https://huggingface.co/TehVenom/Pygmalion-7b-4bit-GPTQ-Safetensors), [Metharme 7B](https://huggingface.co/Neko-Institute-of-Science/metharme-7b), [Metharme 7B 4-bit](https://huggingface.co/TehVenom/Metharme-7b-4bit-32g-GPTQ-Safetensors), [PubMed LLaMA 7B](https://huggingface.co/chaoyi-wu/PMC_LLAMA_7B), [MedAlpaca 7B](https://huggingface.co/medalpaca/medalpaca-7b), [Alpaca Native Enhanced](https://huggingface.co/8bit-coder/alpaca-7b-nativeEnhanced)

1\* This has lighter restrictions than Vicuna and was previously listed in the unrestricted section, but it may trend toward shorter generations than Vicuna.

2\* This is better than AlpacaGPT4 in most areas, especially assistant tasks, but is generally worse for long creative generations.

3\* This may be prone to light restrictions that do not necessarily impact the model's quality. The coherency of the model can initially seem dubious, but it works best when given a good prompt to start with. This 7B model is ideal for storywriting and should be adept at longer generations compared to others in the list.

**13B**

**Models** restricted

**Models** unrestricted

**Models**other

[Vicuna 13B v1.1](https://huggingface.co/eachadea/vicuna-13b-1.1)

[GPT4 x Vicuna](https://huggingface.co/NousResearch/gpt4-x-vicuna-13b)2\*

[WizardLM 13B 1.0](https://huggingface.co/localmodels/WizardLM-13B-1.0)4\*

[Vicuna 13B v1.1 4-bit](https://huggingface.co/localmodels/vicuna-13b-v1.1-4bit/tree/main/gptq)

[GPT4 x Vicuna 4-bit](https://huggingface.co/NousResearch/GPT4-x-Vicuna-13b-4bit)2\*

[WizardLM 13B 1.0 4-bit](https://huggingface.co/localmodels/WizardLM-13B-1.0-4bit/tree/main/gptq)4\*

[StableVicuna](https://huggingface.co/LLMs/Stable-Vicuna-13B)1\*

[GPT4 x Alpaca](https://huggingface.co/chavinlo/gpt4-x-alpaca)3\*

[WizardVicunaLM](https://huggingface.co/junelee/wizard-vicuna-13b)5\*

[StableVicuna 4-bit](https://huggingface.co/localmodels/stable-vicuna-13b-4bit/tree/main/gptq)1\*

[GPT4 x Alpaca 4-bit](https://huggingface.co/anon8231489123/gpt4-x-alpaca-13b-native-4bit-128g)3\*

[WizardVicunaLM 4-bit](https://huggingface.co/localmodels/wizard-vicuna-13b-4bit/tree/main/gptq)5\*

[Baize V2 13B](https://huggingface.co/project-baize/baize-v2-13b) ([4-bit](https://huggingface.co/localmodels/baize-v2-13b-4bit/tree/main/gptq))

[Alpaca Native](https://huggingface.co/chavinlo/alpaca-13b)

[WizardVicunaLM Uncensored](https://huggingface.co/ehartford/Wizard-Vicuna-13B-Uncensored)5\*

[OASST LLaMA](https://huggingface.co/dvruette/oasst-llama-13b-2-epochs) ([4-bit](https://huggingface.co/4bit/oasst-llama13b-4bit-128g))

[LLaMA](https://huggingface.co/elinas/llama-13b-hf-transformers-4.29)

[WizardVicunaLM Uncensored 4-bit](https://huggingface.co/localmodels/wizard-vicuna-13b-uncensored-4bit/tree/main/gptq)5\*

Notable Mention: LLaMA with [AlpacaGPT4 LoRA 13B](https://huggingface.co/LLMs/AlpacaGPT4-LoRA-13B-elina) for longer creative generations.

Extra: [Manticore 13B](https://huggingface.co/openaccess-ai-collective/manticore-13b) ([4-bit](https://huggingface.co/Yhyu13/manticore-13b-gptq-4bit)), [GPT4All 13B snoozy](https://huggingface.co/nomic-ai/gpt4all-13b-snoozy) ([4-bit](https://huggingface.co/localmodels/gpt4all-13b-snoozy-4bit/tree/main/gptq)), [Chronos 13B](https://huggingface.co/elinas/chronos-13b) ([4-bit](https://huggingface.co/Yhyu13/chronos-13b-gptq-4bit)), [Pygmalion 13B](https://huggingface.co/TehVenom/Pygmalion-13b-Merged) ([4-bit](https://huggingface.co/notstoic/pygmalion-13b-4bit-128g)), [Metharme 13B](https://huggingface.co/TehVenom/Metharme-13b-Merged) ([4-bit](https://huggingface.co/TehVenom/Metharme-13b-4bit-GPTQ)), [WizardLM 13B Uncensored](https://huggingface.co/ehartford/WizardLM-13B-Uncensored) ([4-bit](https://huggingface.co/4bit/WizardLM-13B-Uncensored-4bit-128g)), [Vicuna Evol-Instruct](https://huggingface.co/LLMs/Vicuna-EvolInstruct-13B), [LLaVA Delta](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0), [MedAlpaca 13B](https://huggingface.co/medalpaca/medalpaca-13b), [GPT4 x Alpaca Roleplay Merge](https://huggingface.co/teknium/Base-GPT4-x-Alpaca-Roleplay-Lora) ([4-bit V2](https://huggingface.co/teknium/GPT4-x-Alpaca13b-RolePlayLora-4bit-v2)), [pretrained-sft-do2](https://huggingface.co/dvruette/llama-13b-pretrained-sft-do2) ([4-bit](https://huggingface.co/TheYuriLover/llama-13b-pretrained-sft-do2-4bit-128g-TRITON)), [Toolpaca](https://huggingface.co/chavinlo/toolpaca), [Vicuna 13B v0](https://huggingface.co/jeffwan/vicuna-13b) ([4-bit](https://huggingface.co/anon8231489123/vicuna-13b-GPTQ-4bit-128g)), [WizardLM 13B 1.0 diff weights](https://huggingface.co/WizardLM/WizardLM-13B-1.0)

1\* StableVicuna has almost universally higher benchmarks than regular Vicuna, but it fails challenge questions that even Vicuna 7B can answer. It is also based on Vicuna v0. For real usage, its quality seems about on par or slightly worse than Vicuna v1.1.

2\* Not completely unrestricted, and this model fails several logic tests that GPT4 x Alpaca passes. However, it may be better than GPT4 x Alpaca for creative tasks. While its restrictions are almost negligible, it inherits some of Vicuna's inherent limitations. Without proper prompting, this may result in generations with similar plot progressions and endings like ChatGPT, _e.g. "they lived happily ever after"_

3\* The original top choice for weeks and a model that can still be used today for various creative uses. GPT4 x Alpaca naturally produces flowery language that some may consider ideal for storytelling. However, this model may be considered the worst for following complex instructions.

4\* This is an official release from the WizardLM team trained with the full dataset of 250K evolved instructions. It adopts the prompt format from Vicuna v1.1, and this model should be used over the older, experimental WizardVicunaLM.

5\* This is an experimental model designed for proof of concept. It is a combination of WizardLM's dataset, ChatGPT's conversation extension, and Vicuna's tuning method.

**30B**

**Models** restricted

**Models** unrestricted

[Guanaco](https://huggingface.co/timdettmers/guanaco-33b-merged) ([4-bit](https://huggingface.co/MetaIX/Guanaco-33B-4bit))

[GPT4 Alpaca LoRA Merge](https://huggingface.co/MetaIX/GPT4-X-Alpaca-30B-4bit)2\*

[OASST RLHF 2 LLaMA](https://huggingface.co/Yhyu13/oasst-rlhf-2-llama-30b-7k-steps-hf) ([4-bit](https://huggingface.co/Yhyu13/oasst-rlhf-2-llama-30b-7k-steps-gptq-4bit))1\*

[Alpaca LoRA 30B Merge](https://huggingface.co/elinas/alpaca-30b-lora-int4)

[OASST SFT 7 LLaMA 4-bit](https://huggingface.co/MetaIX/OpenAssistant-Llama-30b-4bit)1\*

[LLaMA](https://huggingface.co/elinas/llama-30b-hf-transformers-4.29)

Extra: [WizardLM 30B Uncensored](https://huggingface.co/ehartford/WizardLM-30B-Uncensored), [WizardLM 30B Uncensored 4-bit](https://huggingface.co/localmodels/WizardLM-30B-Uncensored-gptq), [OASST SFT 6 LLaMA 4-bit](https://huggingface.co/MetaIX/OpenAssistant-Llama-30b-4bit/tree/1c2afcb361eb5af8b7e04c3bd53008c55d594f9f)commit 1c2afcb, [OASST RLHF 2 LLaMA XOR](https://huggingface.co/OpenAssistant/oasst-rlhf-2-llama-30b-7k-steps-xor), [OASST SFT 7 LLaMA XOR](https://huggingface.co/OpenAssistant/oasst-sft-7-llama-30b-xor)

1\* This is a finalized version of OASST LLaMA from Open Assistant.

2\* This may be more prone to hallucinatory issues than the original Alpaca LoRA Merge.

**65B**

[Guanaco 65B 4-bit](https://huggingface.co/localmodels/guanaco-65b-gptq)

[LLaMA](https://huggingface.co/elinas/llama-65b-hf-transformers-4.29)

Extra: [LLaMA-Adapter V2 Chat](https://github.com/ZrrSkywalker/LLaMA-Adapter/tree/main/llama_adapter_v2_chat65b)

LoRA
----

Sorted alphabetically:

[Alpaca 7B](https://huggingface.co/tloen/alpaca-lora-7b)

[Alpaca 13B](https://huggingface.co/chansung/alpaca-lora-13b)

[Alpaca 30B](https://huggingface.co/chansung/alpaca-lora-30b)

[Alpaca 65B](https://huggingface.co/chansung/alpaca-lora-65b)

[Alpaca 7B Elina](https://huggingface.co/LLMs/Alpaca-LoRA-7B-elina)\*

[Alpaca 13B Elina](https://huggingface.co/LLMs/Alpaca-LoRA-13B-elina)\*

[Alpaca 30B Elina](https://huggingface.co/LLMs/Alpaca-LoRA-30B-elina)\*

[Alpaca 65B Elina](https://huggingface.co/LLMs/Alpaca-LoRA-65B-elina)\*

[AlpacaGPT4 7B Elina](https://huggingface.co/LLMs/AlpacaGPT4-LoRA-7B-elina)\*

[AlpacaGPT4 13B Elina](https://huggingface.co/LLMs/AlpacaGPT4-LoRA-13B-elina)\*

[Baize 7B](https://huggingface.co/project-baize/baize-lora-7B)

[Baize 7B Healthcare](https://huggingface.co/project-baize/baize-healthcare-lora-7b)

[Baize 13B](https://huggingface.co/project-baize/baize-lora-13B)

[Baize 30B](https://huggingface.co/project-baize/baize-lora-30B)

[gpt4all (7B)](https://huggingface.co/nomic-ai/gpt4all-lora)

[GPT4 Alpaca 7B](https://huggingface.co/chansung/gpt4-alpaca-lora-7b)\*\*

[GPT4 Alpaca 13B](https://huggingface.co/chansung/gpt4-alpaca-lora-13b)\*\*

[GPT4 Alpaca 30B](https://huggingface.co/chansung/gpt4-alpaca-lora-30b)\*\*

[GPT4 Alpaca 65B](https://huggingface.co/chtan/gpt4-alpaca-lora_mlp-65b)\*\*

[GPT4 x Alpaca RP (13B)](https://huggingface.co/ZeusLabs/gpt4-x-alpaca-rp-lora)

[LLaMA Deus V3 (7B)](https://huggingface.co/teknium/llama-deus-7b-v3-lora)

[MedAlpaca 7B](https://huggingface.co/medalpaca/medalpaca-lora-7b-16bit)

[MedAlpaca 13B](https://huggingface.co/medalpaca/medalpaca-lora-13b-8bit)

[MedAlpaca 30B](https://huggingface.co/medalpaca/medalpaca-lora-30b-8bit)

[StackLLaMA 7B](https://huggingface.co/trl-lib/llama-7b-se-rl-peft)

[SuperCOT (7B, 13B, 30B)](https://huggingface.co/kaiokendev/SuperCOT-LoRA)

[Vicuna Evol-Instruct 7B](https://huggingface.co/LLMs/Vicuna-LoRA-EvolInstruct-7B)

[Vicuna Evol-Instruct 13B](https://huggingface.co/LLMs/Vicuna-LoRA-EvolInstruct-13B)

[Vicuna Evol-Instruct Starcoder (13B)](https://huggingface.co/LLMs/Vicuna-LoRA-EvolInstruct-StarCoder)

\*Alpaca LoRA Elina checkpoints are trained with longer cutoff lengths than their original counterparts. AlpacaGPT4 Elina supersedes GPT4 Alpaca.

\*\*GPT4 Alpaca and GPT4 x Alpaca are not the same. GPT4 Alpaca uses the GPT-4 dataset from Microsoft Research.

Other Languages
---------------

Sorted alphabetically:

Chinese Alpaca LoRA ([GitHub](https://github.com/ymcui/Chinese-LLaMA-Alpaca)): [7B](https://huggingface.co/ziqingyang/chinese-alpaca-lora-7b), [13B](https://huggingface.co/ziqingyang/chinese-alpaca-lora-13b)

Chinese ChatFlow ([GitHub](https://github.com/CVI-SZU/Linly)): [7B](https://huggingface.co/Linly-AI/ChatFlow-7B), [13B](https://huggingface.co/Linly-AI/ChatFlow-13B)

Chinese LLaMA Extended ([GitHub](https://github.com/LianjiaTech/BELLE)): [7B](https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-7B), [13B](https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B)

Chinese LLaMA LoRA ([GitHub](https://github.com/ymcui/Chinese-LLaMA-Alpaca)): [7B](https://huggingface.co/ziqingyang/chinese-llama-lora-7b), [13B](https://huggingface.co/ziqingyang/chinese-llama-lora-13b)

Chinese Vicuna LoRA ([GitHub](https://github.com/Facico/Chinese-Vicuna)): [7B](https://huggingface.co/Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco), [13B](https://huggingface.co/Chinese-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco)

French LoRA ([GitHub](https://github.com/bofenghuang/vigogne)): [7B](https://huggingface.co/bofenghuang/vigogne-lora-7b), [13B](https://huggingface.co/bofenghuang/vigogne-lora-13b), [30B](https://huggingface.co/bofenghuang/vigogne-lora-30b)

Italian LoRA ([GitHub](https://github.com/RSTLess-research/Fauno-Italian-LLM/)): [7B](https://huggingface.co/andreabac3/Fauno-Italian-LLM-7B), [13B](https://huggingface.co/andreabac3/Fauno-Italian-LLM-13B)

Japanese LoRA ([GitHub](https://github.com/kunishou/Japanese-Alpaca-LoRA)): [7B](https://huggingface.co/kunishou/Japanese-Alpaca-LoRA-7b-v0), [13B](https://huggingface.co/kunishou/Japanese-Alpaca-LoRA-13b-v0), [30B](https://huggingface.co/kunishou/Japanese-Alpaca-LoRA-30b-v0), [65B](https://huggingface.co/kunishou/Japanese-Alpaca-LoRA-65b-v0)

Korean LoRA ([GitHub](https://github.com/Beomi/KoAlpaca)): [13B](https://huggingface.co/beomi/KoAlpaca-13B-LoRA), [30B](https://huggingface.co/beomi/KoAlpaca-30B-LoRA), [65B](https://huggingface.co/beomi/KoAlpaca-65B-LoRA)

[Portuguese LoRA](https://huggingface.co/22h/cabrita-lora-v0-1) ([GitHub](https://github.com/22-hours/cabrita))

[Russian LoRA 7B Merge](https://huggingface.co/IlyaGusev/llama_7b_ru_turbo_alpaca_lora_merged)

[Russian LoRA 13B](https://huggingface.co/IlyaGusev/llama_13b_ru_turbo_alpaca_lora)

[Spanish LoRA 7B](https://huggingface.co/bertin-project/bertin-alpaca-lora-7b)

llama.cpp
---------

Models that aren't worth including are not listed here.

**Update:** The quantization format has been [updated](https://github.com/ggerganov/llama.cpp/pull/1405). All ggml model files using the old format will not work with the latest llama.cpp code. If you want to use models with the old format, commit cf348a6 is before the breaking change. This list may include a few models in the old format.

Sorted alphabetically:

**7B**

[Alpaca Native](https://huggingface.co/xzuyn/Alpaca-Native-7B-GGML)

[Baize V2 7B](https://huggingface.co/localmodels/baize-v2-7b-4bit/tree/main/ggml)

[Metharme 7B](https://huggingface.co/TehVenom/Metharme-7b-4bit-Q4_1-GGML)

[Pygmalion 7B](https://huggingface.co/TehVenom/Pygmalion-7b-4bit-Q4_1-GGML)

[Vicuna v1.1](https://huggingface.co/CRD716/ggml-vicuna-1.1-quantized)

[WizardLM Uncensored](https://huggingface.co/localmodels/WizardLM-7B-Uncensored-4bit/tree/main/ggml)

Extra or old format: [MedAlpaca](https://huggingface.co/xzuyn/MedAlpaca-7B-GGML), [Vicuna v0](https://huggingface.co/eachadea/legacy-ggml-vicuna-7b-4bit)

**13B**

[Baize V2 13B](https://huggingface.co/localmodels/baize-v2-13b-4bit/tree/main/ggml)

[GPT4All 13B snoozy](https://huggingface.co/localmodels/gpt4all-13b-snoozy-4bit/tree/main/ggml)

[GPT4 x Alpaca](https://huggingface.co/xzuyn/GPT4-x-Alpaca-Native-13B-GGML)

[GPT4 x Vicuna](https://huggingface.co/localmodels/gpt4-x-vicuna-4bit/tree/main/ggml)

[Metharme 13B](https://huggingface.co/TehVenom/Metharme-13b-GGML)

[Pygmalion 13B](https://huggingface.co/TehVenom/Pygmalion-13b-GGML)

[StableVicuna](https://huggingface.co/localmodels/stable-vicuna-13b-4bit/tree/main/ggml)

[Vicuna v1.1](https://huggingface.co/eachadea/ggml-vicuna-13b-1.1)

[WizardLM 13B Uncensored](https://huggingface.co/TehVenom/WizardLM-13B-Uncensored-Q5_1-GGML)

[WizardLM 13B 1.0](https://huggingface.co/localmodels/WizardLM-13B-1.0-4bit/tree/main/ggml)\*

[WizardVicunaLM](https://huggingface.co/localmodels/wizard-vicuna-13b-4bit/tree/main/ggml)\*\*

[WizardVicunaLM Uncensored](https://huggingface.co/localmodels/wizard-vicuna-13b-uncensored-4bit/tree/main/ggml)\*\*

Extra or old format: [Vicuna v0](https://huggingface.co/eachadea/legacy-ggml-vicuna-13b-4bit), [OASST LLaMA](https://huggingface.co/Black-Engineer/oasst-llama13b-ggml-q4), [pretrained-sft-do2](https://huggingface.co/Black-Engineer/llama-13b-pretrained-sft-do2-ggml-q4), [Alpaca Native](https://huggingface.co/Pi3141/alpaca-native-13B-ggml), [Toolpaca](https://huggingface.co/eachadea/ggml-toolpaca-13b-4bit)

\*This is an official release from the WizardLM team trained with the full dataset of 250K evolved instructions. It adopts the prompt format from Vicuna v1.1, and this model should be used over the older, experimental WizardVicunaLM.

\*\*This is an experimental model designed for proof of concept. It is a combination of WizardLM's dataset, ChatGPT's conversation extension, and Vicuna's tuning method.

**30B**

[GPT4 Alpaca LoRA Merge](https://huggingface.co/MetaIX/GPT4-X-Alpaca-30B-4bit)

[Guanaco](https://huggingface.co/MetaIX/Guanaco-33B-4bit)

[OASST SFT 7 LLaMA](https://huggingface.co/MetaIX/OpenAssistant-Llama-30b-4bit)

[SuperCOT](https://huggingface.co/localmodels/SuperCOT-30B-ggml)

[WizardLM 30B Uncensored](https://huggingface.co/localmodels/WizardLM-30B-Uncensored-ggml)

[WizardVicunaLM 30B Uncensored](https://huggingface.co/localmodels/Wizard-Vicuna-30B-Uncensored-ggml)

Extra or old format: [Alpaca LoRA Merge](https://huggingface.co/Pi3141/alpaca-lora-30B-ggml)

**65B**

[Guanaco](https://huggingface.co/localmodels/guanaco-65b-ggml)

[VicUnlocked Alpaca 65B](https://huggingface.co/Aeala/VicUnlocked-alpaca-65b-GGML)

* * *

Prompt Templates
================

For optimal results, you need to use the correct prompt template for the model you're using. This section lists the main prompt templates and some examples of what uses it. This list is not comprehensive.

**Alpaca**

    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
    ### Instruction:
    *your text here*
    
    ### Response:
    

Applies to: Alpaca LoRA, Alpaca Native, GPT4 Alpaca LoRA, GPT4 x Alpaca

**Alpaca with Input**

    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction:
    *your text here*
    
    ### Input:
    *your text here*
    
    ### Response:
    

Applies to: Alpaca LoRA, Alpaca Native, GPT4 Alpaca LoRA, GPT4 x Alpaca

**OpenAssistant LLaMA:**

    <|prompter|>*your text here*<|endoftext|><|assistant|>
    

Applies to: OASST LLaMA 13B, OASST SFT 7 LLaMA, OASST RLHF 2 LLaMA, pretrained-sft-do2

**Vicuna v0**

    A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.
    
    ### Human: *your text here*
    ### Assistant:
    

Applies to: StableVicuna v0, Vicuna v0

**Vicuna v1.1**

    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    
    USER: *your text here*
    ASSISTANT:
    

Applies to: StableVicuna v2, Vicuna Evol-Instruct, Vicuna v1.1, WizardVicunaLM and derivatives

### Other Templates

**GPT4 x Vicuna:**

    ### Instruction:
    *your text here*
    
    ### Response:
    

or

    ### Instruction:
    *your text here*
    
    ### Input:
    *your text here*
    
    ### Response:
    

**Guanaco QLoRA**\*

    ### Human: *your text here*
    
    ### Assistant:
    

\*This should not be confused with the older Guanaco model made by a separate group and using a different dataset.

**Metharme and Pygmalion**

[Metharme explanation](https://huggingface.co/PygmalionAI/metharme-7b#prompting)

[Pygmalion explanation](https://huggingface.co/PygmalionAI/pygmalion-7b#prompting)

**WizardLM 7B**

    *your text here*
    
    ### Response:
    

**WizardLM 13B 1.0**\*

    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: *your text here* ASSISTANT:
    

\*This should not be confused with the older WizardLM models that use the dataset of 70K evolved instructions.
