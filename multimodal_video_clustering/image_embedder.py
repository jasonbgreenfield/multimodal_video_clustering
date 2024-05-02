import os
import re
from PIL import Image

import clip
import torch
from tqdm import tqdm


__all__ = ['ImageEmbedder']


class ImageEmbedder(object):
    """
    Class for embedding images
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def _get_frame_names(self):
        """
        get list of image frames in data_dir
        """
        # init regex for frame file names
        frame_temp = r'.+-Scene-\d+-\d+\.jpg'
        # get name of all scenes
        frame_file_names = [x for x in os.listdir(self.data_dir) if re.match(frame_temp, x)]
        return frame_file_names

    def _save_embedding(self, frame_name, img_embedding):
        """
        save image embedding tensor to .pt file
        :param frame_name: name of image that was embedded
        :param img_embedding: embedding tensor to save
        """
        # format file name
        file_name = frame_name.replace('.jpg', '_clip_image_embedding.pt')
        # save data
        torch.save(img_embedding, os.path.join(self.data_dir, file_name))

    def embed_images(self):
        """
        read in video files and detect scenes
        """
        for frame_name in tqdm(self._get_frame_names()):
            # read image
            clip_img = self.preprocess(Image.open(os.path.join(self.data_dir, frame_name))).unsqueeze(0).to(self.device)
            # generate embedding
            with torch.no_grad():
                clip_image_features = self.clip_model.encode_image(clip_img)
            # save embedding
            self._save_embedding(frame_name, clip_image_features)

