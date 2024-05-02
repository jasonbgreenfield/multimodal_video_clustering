import csv
import os
import re
import shutil
from tqdm import tqdm
import warnings

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

# Suppress the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


__all__ = ['Clusterer']


class Clusterer(object):
    """
    Class for clustering multi-modal audio, text and image embeddings
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_dir_files = [x for x in os.listdir(self.data_dir)]

    def _get_video_embeddings(self):
        """
        read in previously generated embeddings, standardize each embedding and concatenate into mult-modal embedding
        :return: dict mapping video name to multi-modal video embedding
        """
        # init regex for scene file names
        scene_temp = r'(.+-Scene-\d+)\.mp4'
        # get name of all videos that are not scenes
        video_file_names = [x for x in os.listdir(self.data_dir) if '.mp4' in x and not re.match(scene_temp, x)]
        videos = {}
        ###
        # load all embeddings (skip if any embedding is missing)
        ###
        for video_name in tqdm(video_file_names):
            video_name_full_path = os.path.join(self.data_dir, video_name)
            # 0. AudioCLIP audio embedding
            fn_audioclip = video_name_full_path.replace('.mp4', '_audioclip_audio_embedding.pt')
            if not os.path.exists(fn_audioclip):
                continue
            audioclip_audio_embedding = torch.load(fn_audioclip)
            # 1. Whisper audio embedding
            fn_whisper1 = video_name_full_path.replace('.mp4', '_whisper_audio_embedding.pt')
            if not os.path.exists(fn_whisper1):
                continue
            whisper_audio_embedding = torch.load(fn_whisper1)
            # 2. Clip image embeddings (avg across all frames for video
            clip_image_embedding_fns = [os.path.join(self.data_dir, x) for x in os.listdir(self.data_dir) if video_name.replace('.mp4', '') in x and '_clip_image_embedding.pt' in x]
            if len(clip_image_embedding_fns) == 0:
                continue
            clip_image_embedding_tensors = [torch.load(x) for x in clip_image_embedding_fns]
            # Convert list of tensors to a single tensor and sum them
            sum_image_embedding_tensors = torch.stack(clip_image_embedding_tensors).sum(dim=0)
            # Calculate the average
            clip_image_embedding_avg = sum_image_embedding_tensors / len(clip_image_embedding_tensors)
            # 3. RoBERTa text embedding description
            description_fn = video_name_full_path.replace('.mp4', '_roberta_description_text_embedding.pt')
            if not os.path.exists(description_fn):
                continue
            roberta_description_text_embedding = torch.load(description_fn)
            # 4. RoBERTa text embedding ocr
            ocr_fn = video_name_full_path.replace('.mp4', '_roberta_ocr_text_embedding.pt')
            if not os.path.exists(ocr_fn):
                continue
            roberta_ocr_text_embedding = torch.load(ocr_fn)
            ###
            # Flatten each tensor to a single dimension
            ###
            audioclip_audio_embedding_flat = audioclip_audio_embedding.flatten()
            whisper_audio_embedding_flat = whisper_audio_embedding.flatten()
            clip_image_embedding_avg_flat = clip_image_embedding_avg.flatten()
            roberta_description_text_embedding_flat = roberta_description_text_embedding.flatten()
            roberta_ocr_text_embedding_flat = roberta_ocr_text_embedding.flatten()
            ###
            # Convert size of each flattened tensor to about 2048
            # by either interpolating to increase size or down sampling to decrease size
            ###
            def resize_tensor(tensor_to_resize):
                # no resize necessary
                if 1948 < len(tensor_to_resize) < 2148:
                    resized_tensor = tensor_to_resize
                # interpolate -- increase size
                elif len(tensor_to_resize) < 2048:
                    # Reshape the tensor to have shape (1, 1, length)
                    input_tensor = tensor_to_resize.unsqueeze(0).unsqueeze(0)
                    # Use interpolate to increase the length
                    interpolated_tensor = F.interpolate(input_tensor, size=(2048), mode='linear', align_corners=False)
                    # Reshape the interpolated tensor back to 1D
                    resized_tensor = interpolated_tensor.squeeze()
                # down sample -- decrease size
                elif len(tensor_to_resize) > 2048:
                    # Determine the reduction factor
                    reduction_factor = len(tensor_to_resize) // 2048
                    # Downsample the tensor using average pooling
                    resized_tensor = F.avg_pool1d(tensor_to_resize.unsqueeze(0).unsqueeze(0),
                                                  kernel_size=reduction_factor).squeeze()
                # pad to length 2100
                # Calculate the padding length
                padding_length = 2100 - len(resized_tensor)
                # Pad the resized tensor with zeros
                padded_tensor = torch.cat((resized_tensor, torch.zeros(padding_length)), dim=0)
                # return resized and padded tensor
                return padded_tensor
            audioclip_audio_embedding_flat_resized = resize_tensor(audioclip_audio_embedding_flat)
            whisper_audio_embedding_flat_resized = resize_tensor(whisper_audio_embedding_flat)
            clip_image_embedding_avg_flat_resized = resize_tensor(clip_image_embedding_avg_flat)
            roberta_description_text_embedding_flat_resized = resize_tensor(roberta_description_text_embedding_flat)
            roberta_ocr_text_embedding_flat_resized = resize_tensor(roberta_ocr_text_embedding_flat)
            ###
            # Concatenate tensors to make joint embedding
            ###
            # Concatenate the tensors
            concatenated_tensor = torch.cat((audioclip_audio_embedding_flat_resized,
                                             whisper_audio_embedding_flat_resized,
                                             clip_image_embedding_avg_flat_resized,
                                             roberta_description_text_embedding_flat_resized,
                                             roberta_ocr_text_embedding_flat_resized), dim=0)
            # save joint embeddings
            videos[video_name] = concatenated_tensor
        print(f'got joint embeddings for {len(videos)}/{len(video_file_names)} videos')
        # return dict mapping video name to video embeddings
        return videos

    def _save_embeddings(self, video_embeddings):
        """
        save each multi-modal embedding as its own .pt file
        :param video_embeddings: dict mapping video file names to video embeddings
        """
        for video_name, video_embedding in video_embeddings.items():
            # format file name
            file_name = f'{video_name}_video_embedding.pt'
            # save data
            torch.save(video_embedding, os.path.join(self.data_dir, file_name))

    def _elbow_method(self, embeddings_standardized):
        """
        generate plot showing inertia for various k values
        :param embeddings_standardized: list of standardized embeddings for each video
        """
        # Initialize an empty list to store inertia values
        inertia = []
        # Define the range of k values to try
        k_values = range(1, 25)
        # Run KMeans for each k value and compute inertia
        for k in tqdm(k_values):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(embeddings_standardized)
            inertia.append(kmeans.inertia_)  # Inertia is same as distortion
        # Plot the elbow curve with Distortion
        plt.plot(k_values, inertia, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title(f'Elbow Method for KMeans Clustering (Inertia)')
        plt.xticks(k_values)
        plt.grid(True)
        # Save the plot as PNG
        plt.savefig(os.path.join(self.data_dir, f'elbow_method_inertia.png'))

    def _save_clusters(self, video_embeddings, kmeans_model):
        """
        save csv mapping video names to clusters and copy each video to /clusters/ dir in new dir for its cluster
        :param video_embeddings: dict mapping video file names to video embeddings
        :param kmeans_model: model with kmeans cluster labels
        """
        # init cluster csv fn
        clusters_fn = os.path.join(self.data_dir, 'clusters.csv')
        cluster_ids = [['video_name', 'cluster_id']]
        # group videos into assigned clusters
        for video_name, cluster_id in zip(video_embeddings, kmeans_model.labels_):
            # save video and cluster data
            cluster_ids.append([video_name, cluster_id])
            # make cluster dir
            cluster_dir = f'clusters/cluster_{cluster_id}'
            full_cluster_dir = os.path.join(self.data_dir, cluster_dir)
            os.makedirs(full_cluster_dir, exist_ok=True)
            # get video file name
            video_file_name = video_name
            # format full path for old file
            old_video_file_name = os.path.join(self.data_dir, video_file_name)
            # format full path for new file
            new_video_file_name = os.path.join(full_cluster_dir, video_file_name)
            # copy video to cluster dir
            with open(old_video_file_name, 'rb') as f_in:
                with open(new_video_file_name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        # save csv mapping video names to cluster ids
        with open(clusters_fn, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(cluster_ids)

    def cluster(self):
        """
        generate multi-modal embeddings, run elbow method to get optimal number of clusters, run clustering
        """
        # get dict of video names -> joint embeddings
        video_embeddings = self._get_video_embeddings()
        # save scene embeddings
        self._save_embeddings(video_embeddings)
        # standardize embeddings
        embeddings_standardized = StandardScaler().fit_transform([y.detach().numpy() for x, y in video_embeddings.items()])
        # run elbow method to get optimal number of clusters
        self._elbow_method(embeddings_standardized)
        # fit model
        num_clusters = 10
        kmeans_model = KMeans(n_clusters=num_clusters).fit(embeddings_standardized)
        # write clusters to output dir
        self._save_clusters(video_embeddings, kmeans_model)
