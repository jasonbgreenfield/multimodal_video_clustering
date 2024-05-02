from cluster_videos import VideoSplitter, AudioEmbedder, ImageEmbedder, TextEmbedder, Clusterer

# init path to video data
data_dir = '/full/path/to/videos'

# 0. split videos into scenes and image frames for each scene
# data in: .mp4 video files
# data out: a series of scene videos which cumulatively make up the original video and images sampled from each scene
# Note: images from scenes are used for image embeddings and ocr text extraction
video_splitter = VideoSplitter(data_dir)
video_splitter.split_videos_into_scenes()

# 1. embed audio for each video
# data in: audio wave extracted from each full video
# models: AudioCLIP, and Whisper
# data out, AudioCLIP: audio features from pretrained AudioCLIP model based on spectrogram
# data out, Whisper: audio features from pretrained Whisper model
# Note: load AudioCLIP from: https://github.com/AndreyGuzhov/AudioCLIP/blob/master/assets/AudioCLIP-Full-Training.pt
audioclip_model_path = '/full/path/to/audioclipmodel'
audio_embedder = AudioEmbedder(data_dir, audioclip_model_path)
audio_embedder.embed_audio()

# 2. embed each frame image for each scene
# model: CLIP
# data in: images from each scene
# data out: averaged CLIP embedding for all frames in a full video
image_embedder = ImageEmbedder(data_dir)
image_embedder.embed_images()

# 3. embed text from each scene
# model: RoBERTa for text embedding, easyocr package for ocr text extraction
# data in: images from each scene and description of video
# data out: RoBERTa embeddings, one for ocr text and one for video description
text_embedder = TextEmbedder(data_dir)
text_embedder.embed_text()

# 4. cluster joint embeddings and save scenes to cluster dirs
# model: KMeans
# data in: audio embeddings, image embeddings, text embeddings
# data out: standardized embeddings and cluster labels
clusterer = Clusterer(data_dir)
clusterer.cluster()
