import logging
import tarfile
import os
from shutil import which


def make_tarfile(archive_name, video_folder):
    logging.info(f'Compressing synthetic videos into tarball {archive_name}.tar.gz')

    with tarfile.open(f'{archive_name}.tar.gz', 'w:gz') as tar:
        tar.add(video_folder, arcname=os.path.basename(video_folder))

    logging.info(f'Tarfile can be found at {archive_name}.tar.gz')

def check_blender_install(blender):
        # Check for Blender installation
    if which(blender) is None:
        raise Exception('--blender does not point to a (working) Blender installation. You can use ./install_blender.sh to install Blender.')
