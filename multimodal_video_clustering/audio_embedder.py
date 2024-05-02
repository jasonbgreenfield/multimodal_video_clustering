import os
import re
import warnings

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, WhisperProcessor

from AudioCLIP.model import AudioCLIP
from AudioCLIP.utils.transforms import ToTensor1D

# filter warnings
warnings.filterwarnings("ignore", category=UserWarning, message="PySoundFile failed.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.*")

__all__ = ['AudioEmbedder']


class AudioEmbedder(object):
    """
    Class for embedding audio
    """

    def __init__(self, data_dir, audioclip_model_path=None):
        self.data_dir = data_dir
        self.audio_clip_model_path = audioclip_model_path
        # NOTE: there are two, one from each tutorial above, we should test each accordingly
        self.processor_auto = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        self.processor_whisp = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

    def _get_videos(self):
        """
        get list of videos in data_dir
        """
        # init regex for scene file names
        scene_temp = r'(.+-Scene-\d+)\.mp4'
        # get name of all videos that are not scenes
        video_file_names = [x for x in os.listdir(self.data_dir) if '.mp4' in x and not re.match(scene_temp, x)]
        return video_file_names

    def _get_audio_wave(self, scene_name):
        """
        load audio file
        :param scene_name: name of audio file
        :return: audio wave and sampling rate
        """
        # format file name
        video_file_name = os.path.join(self.data_dir, scene_name)
        # load video with librosa
        y, sr = librosa.load(video_file_name)
        return y, sr

    def _embed_audioclip(self, video_name, y):
        """
        embed audio using AudioCLIP model
        :param video_name: file name for video whose audio is being embedded
        :param y: audio wave
        """
        # init output fn and skip if embedding is already processed
        fn_audioclip = video_name.replace('.mp4', '_audioclip_audio_embedding.pt')
        full_fn_audioclip = os.path.join(self.data_dir, fn_audioclip)
        if os.path.exists(full_fn_audioclip):
            return
        # load model
        aclp = AudioCLIP(pretrained=self.audio_clip_model_path)
        # load audio transforms
        audio_transforms = ToTensor1D()
        # compute spectrograms using trained audio-head (fbsp-layer of ESResNeXt)
        # thus, the actual time-frequency representation will be visualized
        spec = aclp.audio.spectrogram(torch.from_numpy(y.reshape(1, 1, -1)))
        spec = np.ascontiguousarray(spec.detach().numpy()).view(np.complex64)
        pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()
        # model requires batches to pass redundant copies of input audio as placeholder
        audio = [[y, pow_spec], [y, pow_spec]]
        # Input preparation
        # AudioCLIP handles raw audio on input, so the input shape is [batch x channels x duration]
        audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio])
        # get embeddings
        # AudioCLIP's output: Tuple[Tuple[Features, Logits], Loss]
        ((audio_features, _, _), _), _ = aclp(audio=audio)
        # normalize embeddings
        audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
        # only keep first embeddings since we ran two duplicate ones
        audio_features = audio_features[0]
        # save embeddings
        torch.save(audio_features, full_fn_audioclip)

    def _embed_whisper(self, video_name, y, sr):
        """
        embed audio using Whisper model
        :param video_name: file name for video whose audio is being embedded
        :param y: audio wave
        :param sr: sampling rate for audio file
        """
        # resample audio if necessary since the processors require 16k
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        # init output fn and skip if embedding is already processed
        fn_whisper1 = video_name.replace('.mp4', '_whisper_audio_embedding.pt')
        full_fn_whisper1 = os.path.join(self.data_dir, fn_whisper1)
        if not os.path.exists(full_fn_whisper1):
            # generate embedding
            inputs = self.processor_auto(y, return_tensors="pt", return_attention_mask=True, sampling_rate=16000)
            torch.save(inputs['input_features'], full_fn_whisper1)

    def embed_audio(self):
        """
        generate audio embeddings for each video using both AudioCLIP and Whisper
        :return:
        """
        for video_name in tqdm(self._get_videos()):
            # get audio from video
            try:
                audio_wave, audio_sampling_rate = self._get_audio_wave(video_name)
            except Exception as e:
                print(f'exception with {video_name}\n{e}')
            # embed audio with AudioCLIP
            self._embed_audioclip(video_name, audio_wave)
            # embed audio with Whisper
            self._embed_whisper(video_name, audio_wave, audio_sampling_rate)

