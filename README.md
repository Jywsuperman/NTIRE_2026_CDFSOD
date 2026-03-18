# 🥇 NTIRE 2026 CD-FSOD Challenge @ CVPR Workshop

We are the **award-winning team** of the **NTIRE 2026 Cross-Domain Few-Shot Object Detection (CD-FSOD) Challenge** at the **CVPR Workshop**.

- 🏆 **Track**: `open-source track`
- 🎖️ **Award**: **2nd Place**

🔗 [NTIRE 2026 Official Website](https://cvlai.net/ntire/2026/)  
🔗 [NTIRE 2026 Challenge Website](https://www.codabench.org/competitions/12873/)  
🔗 [NTIRE2026_CDFSOD Challenge Repository](https://github.com/ohMargin/NTIRE2026_CDFSOD)

![CD-FSOD Task](https://upload-images.jianshu.io/upload_images/9933353-3d7be0d924bd4270.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

## 🧠 Overview

This repository contains our solution for the `open-source track` of the NTIRE 2026 CD-FSOD Challenge.  
We propose a **Pseudo-Label Driven Vision-Language Grounding framework** for CD-FSOD, which combines large-scale vision-language foundation models with an iterative pseudo-labeling strategy.

![Overview](xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)

---

## 🛠️ Environment Setup

***Environment*** This repository requires **PyTorch >= 1.9** and **torchvision**. We recommend using Docker to set up the environment. 
You may use one of the following pre-built Docker images depending on your GPU.

`docker pull pengchuanzhang/maskrcnn:ubuntu18-py3.7-cuda10.2-pytorch1.9`

`docker pull pengchuanzhang/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9`

Then install the following packages:

```
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo
pip install transformers 
python setup.py build develop --user
```


## 📂 Dataset Preparation
Please follow the instructions in the [official CD-FSOD repo](https://github.com/lovelyqian/NTIRE2025_CDFSOD) to download and prepare the dataset.

## Model Zoo

Download the [GLIP pretrained weight](https://huggingface.co/GLIPModel/GLIP/blob/main/glip_large_model.pth) to the `/GLIP/weights` folder.

For more details, please refer to [microsoft/GLIP: Grounded Language-Image Pre-training](https://github.com/microsoft/GLIP).

The released fine-tuned checkpoints are available here: [Fine-tuned model weights](https://huggingface.co/Jelsuperman/CDFSOD/tree/main).

At present, we release the best checkpoints for **Dataset 2** under the **1-shot**, **5-shot**, and **10-shot** settings. Checkpoints for other settings may be released later.

### GroundingDINO Results

The GroundingDINO-based results for **Dataset 1** and **Dataset 3** are available in the following repository:

[GroundingDINO Results Repository](github.com/z-yaz/CDiscover)

## 🔍 Inference & Evaluation

Run the following command to evaluate on the dataset. Set `{config_file}`, `{model_checkpoint}` according to the `Model Zoo`. Set `{odinw_configs}` to the path of the task yaml.


```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=65521 tools/test_grounding_net.py \
      --config-file {config_file} \
      --weight {model_checkpoint} \
      --task_config {odinw_configs} \
      TEST.IMS_PER_BATCH 8 SOLVER.IMS_PER_BATCH 8 \
      TEST.EVAL_TASK detection \
      DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
      DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False \
      DATASETS.USE_OVERRIDE_CATEGORY True \
      DATASETS.USE_CAPTION_PROMPT True


######### for example
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=65521 tools/test_grounding_net.py \
      --config-file /home/CDFSOD/GLIP/configs/pretrain/glip_Swin_L.yaml \
      --weight /home/CDFSOD/GLIP/OUTPUT/1-shot-dataset2.pth \
      --task_config /home/CDFSOD/GLIP/dataset2.yaml \
      TEST.IMS_PER_BATCH 8 SOLVER.IMS_PER_BATCH 8 \
      TEST.EVAL_TASK detection \
      DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
      DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False \
      DATASETS.USE_OVERRIDE_CATEGORY True \
      DATASETS.USE_CAPTION_PROMPT True
```

## Fine-Tuning

Due to the randomness of GLIP, fine-tuning results may vary across runs. We recommend trying multiple random seeds to obtain the best performance. Before each run, please clean up the `OUTPUT` directory, since multiple checkpoints and evaluation files will be saved during training. In our experiments, we use **4 × vGPU-32GB** for fine-tuning.

The prediction JSON files are usually saved in `OUTPUT/eval`, and the corresponding checkpoints are stored in `OUTPUT/ft_task_{N}`. Please back up and remove the existing files in these directories before starting a new experiment.

### Iterative Pseudo-Labeling Pipeline

Our method is based on a multi-stage pseudo-labeling pipeline.

1. **Prepare images for pseudo labeling**  
   Convert the images used for pseudo-label generation into an empty COCO-format annotation file containing only `images` and `categories`.

2. **Generate predictions for pseudo-label candidates**  
   Use a fine-tuned GLIP checkpoint to perform inference on these images and save the predicted bounding boxes in JSON format.

3. **Filter predictions and build pseudo labels**  
   Apply confidence thresholding and class-wise NMS to the predicted boxes, then convert the remaining predictions into COCO-style pseudo annotations.

4. **Merge pseudo labels with few-shot annotations**  
   Combine the pseudo annotations with the original few-shot ground-truth annotations to form the training annotations for the next round.

5. **Fine-tune on the merged annotations**  
   Use the merged annotation file for the next round of fine-tuning.

This process can be repeated for multiple rounds if needed. The scripts for pseudo-label preparation and pseudo-annotation construction are provided in `tools/pseudo_label/`, and the detailed usage can be found in `tools/pseudo_label/README.md`.

## Acknowledgement

We express our gratitude to the [GLIP](https://github.com/microsoft/GLIP) and [Open-GroundingDINO](https://github.com/longzw1997/Open-GroundingDino) authors for their open-source contribution.

## 📄 Citation
If you use our method or codes in your research, please cite:
```
@inproceedings{qiu2026ntire, 
  title={NTIRE 2026 challenge on cross-domain few-shot object detection: methods and results},
  author={ Qiu, Xingyu and Fu, Yuqian and Jiawei, Geng and Ren, Bin and Jiancheng Pan and Fu, Yanwei and Timofte, Radu and others},
  booktitle={CVPRW},
  year={2026}
}
```





