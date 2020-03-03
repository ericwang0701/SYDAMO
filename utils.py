import logging
import tarfile
import os

def make_tarfile(archive_name, video_folder):
    logging.info(f'Compressing synthetic videos into tarball {archive_name}.tar.gz')

    with tarfile.open(f'{archive_name}.tar.gz', 'w:gz') as tar:
        tar.add(video_folder, arcname=os.path.basename(video_folder))
