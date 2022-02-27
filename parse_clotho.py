import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import librosa
import numpy as np
import pandas as pd


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/clotho/oscar_split_{clip_model_name}_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, jit=False)


    # data = pd.read_csv('./data/clotho_captions_development.csv')
    data = pd.read_csv(args.captions_path)
    print("%0d captions loaded from csv " % len(data))
    all_embeddings = []
    all_captions = []
    for index, row in data.iterrows():

        img_name = row["file_name"]
        # filename = f"./data/development/{img_name}"
        filename = f"{args.audio_path}/{img_name}"

        #compute melspec
        try:
            wav = librosa.load(filename, sr=44100)[0]
            melspec = librosa.feature.melspectrogram(
                wav,
                sr=44100,
                n_fft=2560,
                hop_length=694,
                n_mels=128,
                fmin=20,
                fmax=22050)
            logmel = librosa.core.power_to_db(melspec)
        except ValueError:
            print('ERROR IN:', filename)
        logmel = logmel.astype(np.uint8)
        # image = io.imread(logmel)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        # image = preprocess(Image.fromarray(logmel)).unsqueeze(0)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        data["clip_embedding"] = index
        all_embeddings.append(prefix)
        all_captions.append(data)
        if (index + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--captions_path', default=None)
    parser.add_argument('--audio_path', default=None)
    args = parser.parse_args()
    exit(main(args.clip_model_type))
