import sys
from setuptools import setup

if sys.version_info[0] != 3:
    raise RuntimeError('Unsupported python version "{0}"'.format(
        sys.version_info[0]))


def _get_file_content(file_name):
    with open(file_name, 'r') as file_handler:
        return str(file_handler.read())


def get_long_description():
    return _get_file_content('README.md')


INSTALL_REQUIRES = [
    'clip',
    'easyocr',
    'librosa',
    'numpy',
    'opencv-python',
    'pillow',
    'pytorch-ignite',
    'scenedetect',
    'sklearn',
    'termcolor',
    'torch',
    'torchvision',
    'transformers',
    'visdom'
]

setup(
    name="multimodal_video_clustering",
    version='0.0.1',
    author="Jason Greenfield",
    description="multimodal_video_clustering is a python library for clustering multimodal video embeddings (image, text, audio)",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    keywords='',
    url="https://github.com/jasonbgreenfield/multimodal_video_clustering",
    packages=['multimodal_video_clustering'],
    license="MIT",
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=INSTALL_REQUIRES
)
