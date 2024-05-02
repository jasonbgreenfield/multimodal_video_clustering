from multiprocess import Pool
import os
import re
from tqdm import tqdm

from scenedetect import detect, ContentDetector, open_video, save_images, split_video_ffmpeg


__all__ = ['VideoSplitter']


class VideoSplitter(object):
    """
    Class for splitting videos into individual scenes and sampling frames from each scene
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def _get_videos(self):
        """
        get list of videos in data_dir
        """
        # init regex for scene file names
        scene_temp = r'(.+-Scene-\d+)\.mp4'
        # get name of all videos that are not scenes
        video_file_names = [x for x in os.listdir(self.data_dir) if '.mp4' in x and not re.match(scene_temp, x)]
        return video_file_names

    def split_video(self, video_path):
        """
        split individual video
        :param video_path: path to video to split
        """
        try:
            # detect scenes
            scene_list = detect(os.path.join(self.data_dir, video_path), ContentDetector())
            # split video into scenes and save as individual files
            split_video_ffmpeg(os.path.join(self.data_dir, video_path), scene_list)
            # open video stream
            video_stream = open_video(os.path.join(self.data_dir, video_path))
            # make image files for each scene
            save_images(scene_list, video_stream, num_images=5)
        except Exception as e:
            print(f'split_video exception with {video_path}\n{e}')

    def split_videos_into_scenes(self):
        """
        read in video files and detect scenes, save entire scene as mp4 and five frames as jpg
        """
        # split every video in the input directory
        with Pool(32) as p:
            # Initialize tqdm with the total number of tasks
            with tqdm(total=len(self._get_videos())) as pbar:
                # Use imap to iterate over results asynchronously
                for result in p.imap(self.split_video, self._get_videos()):
                    # Update the progress bar for each completed task
                    pbar.update(1)
