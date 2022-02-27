## Work based on official implementation for the paper ["ClipCap: CLIP Prefix for Image Captioning"](https://arxiv.org/abs/2111.09734)


## Description  
Image captioning is a complicated task, where usually a pretrained detection network is used, requires additional supervision in the form of object annotation. We present a new approach that does not requires additional information (i.e. requires only images and captions), thus can be applied to any data. In addition, our model's training time is much faster than similar methods while achieving comparable to state-of-the-art results, even for the Conceptual Captions dataset contains over 3M images. 

In our work, we use the [CLIP](https://github.com/openai/CLIP) model, which was already trained over an extremely large number of images, thus is capable of generating semantic encodings for arbitrary images without additional supervision. To produce meaningful sentences we fine-tune a pretrained language model, which has been proven to be successful for other natural language tasks. The key idea is to use the CLIP encoding as a prefix to the textual captions by employing a simple mapping network over the raw encoding, and then fine-tune our language model to generate a valid caption. In addition, we present another variant, where we utilize a transformer architecture for the mapping network and avoid the fine-tuning of GPT-2. Still, our light model achieve comaparable to state-of-the-art over nocaps dataset.

## Training prerequisites

[comment]: <> (Dependencies can be found at the [Inference notebook]&#40;https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing&#41; )
Clone, create environment and install dependencies:  
```
git clone https://github.com/anushkajj/ClipAudioCaption.git && cd ClipAudioCaption
conda env create -f environment.yml
conda activate clip_prefix_caption
```

## Clotho training

Download [train_audio and captions](https://zenodo.org/record/3490684#.Yhtnve5Bw-Q) to `data`.

Download model weights to root directory
```
gdown --id 14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT -O model_wieghts.pt 
```
Extract CLIP features using (output is `data/clotho/oscar_split_ViT-B_32_train.pkl`):
```
python parse_clotho.py --clip_model_type ViT-B/32 --caption_path ./data/clotho_captions_development.csv --audio_path ./data/development
```
Train with fine-tuning of GPT2
```
python train.py --data ./data/clotho/oscar_split_ViT-B_32_train.pkl --out_dir ./clotho_train/

```

