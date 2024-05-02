import json
import os
import re

import easyocr
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel


__all__ = ['TextEmbedder']


class TextEmbedder(object):
    """
    Class for embedding text
    """

    def __init__(self, data_dir, model_name='roberta-base'):
        self.data_dir = data_dir
        # load dict mapping video name to video description
        with open(os.path.join(self.data_dir, 'video_descriptions.json')) as f:
            descriptions = json.load(f)
        self.descriptions = descriptions
        # load ocr model
        self.ocr_reader = easyocr.Reader(['en'])
        # Load the RoBERTa model and tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)

    def _get_videos(self):
        """
        get list of videos in data_dir
        """
        # init regex for scene file names
        scene_temp = r'(.+-Scene-\d+)\.mp4'
        # get name of all videos that are not scenes
        video_file_names = [x for x in os.listdir(self.data_dir) if '.mp4' in x and not re.match(scene_temp, x)]
        return video_file_names

    def _get_frame_names(self):
        """
        get list of image frames in data_dir
        """
        # init regex for frame file names
        frame_temp = r'.+-Scene-\d+-\d+\.jpg'
        # get name of all scenes
        frame_file_names = [x for x in os.listdir(self.data_dir) if re.match(frame_temp, x)]
        return frame_file_names

    def _get_ocr_text(self, video_frames):
        """
        extract text from images with ocr
        :param video_frames: list of file names with image frames for this video
        :return: joined text across all frames for this video
        """
        ocr_text = []
        # run ocr on each frame
        for frame_name in video_frames:
            ocr_text = self.ocr_reader.readtext(os.path.join(self.data_dir, frame_name), detail=0)
        # return full text from all frames
        return ' '.join(ocr_text)

    def _embed_text(self, text, output_fn):
        """
        make RoBERTa embedding of text
        :param text: text to embed
        :param output_fn: fn to save embedded text to
        """
        # encode text
        encoded = self.tokenizer.encode_plus(text, padding='max_length', max_length=128, truncation=True,
                                             return_tensors='pt')
        # Concatenate the list of encoded tensors into a single tensor
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        with torch.no_grad():
            model_output = self.model(input_ids, attention_mask=attention_mask)
        # Extract the embeddings from the model output as Tensor
        embedding = model_output.last_hidden_state.mean(dim=1)
        # save embeddings
        torch.save(embedding, os.path.join(self.data_dir, output_fn))

    def embed_text(self):
        """
        make text embeddings for each video using both video description and ocr extracted text
        """
        # get all scenes
        frame_names = self._get_frame_names()
        # embed text for each video
        for video_name in tqdm(self._get_videos()):
            # get description
            if video_name not in self.descriptions:
                print(f'skipping missing description: {video_name}')
            else:
                description_text = self.descriptions[video_name]
                description_fn = video_name.replace('.mp4', '_roberta_description_text_embedding.pt')
                self._embed_text(description_text, description_fn)
            # get ocr text
            video_frames = [x for x in frame_names if video_name in x]
            ocr_text = self._get_ocr_text(video_frames)
            ocr_fn = video_name.replace('.mp4', '_roberta_ocr_text_embedding.pt')
            self._embed_text(ocr_text, ocr_fn)

