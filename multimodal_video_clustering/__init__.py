"""
initialize the classes and set the __all__ params
"""
from multimodal_video_clustering.video_splitter import VideoSplitter
from multimodal_video_clustering.audio_embedder import AudioEmbedder
from multimodal_video_clustering.image_embedder import ImageEmbedder
from multimodal_video_clustering.text_embedder import TextEmbedder
from multimodal_video_clustering.clusterer import Clusterer

__all__ = ['VideoSplitter', 'AudioEmbedder', 'ImageEmbedder', 'TextEmbedder', 'Clusterer']


__version__ = '0.0.1'
__author__ = 'Jason Greenfield'
