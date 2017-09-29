import os
import errno
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from .utils import download_url, check_integrity



class HPatchesSeq(data.Dataset):
    """`HPatches: A benchmark and evaluation of handcrafted and learned local descriptors <https://hpatches.github.io/>`_ Dataset.


    Args:
        root (string): Root directory where images are.
        split_name (string): Name of the train-test split to load.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    urls = {
        'hpatches-seq': [
            'http://www.iis.ee.ic.ac.uk/~vbalnt/hpatches/hpatches-sequences-release.tar.gz',
            'hpatches-sequences-release.tar.gz',
            'c9d94e72b85182ba744534db0aca9c94'
        ],
          'splits': [
            'https://raw.githubusercontent.com/hpatches/hpatches-benchmark/master/python/utils/splits.json',
            'splits.json',
            'b08cae8889120e339f5512c10fad4d7f'
            ]
    }

    image_ext = 'ppm'
    
    def __init__(self, root, split_name, train=True, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.name = 'hpatches-sequences-release'
        self.split_name = split_name
        self.data_dir = os.path.join(self.root, self.name)
        self.data_down = os.path.join(self.root, '{}.tar.gz'.format(self.name))
        self.train = train
        if self.train:
            self.data_file = os.path.join(self.root, '{}.pt'.format(self.name + '_' + self.split_name + '_train' ))
        else:
            self.data_file = os.path.join(self.root, '{}.pt'.format(self.name + '_' + self.split_name + '_test' ))            
        self.transform = transform


        if download:
            self.download()

        if not self._check_datafile_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # load the serialized data
        self.ref_imgs, self.warped_imgs, self.homographies =  torch.load(self.data_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (data1, data2, matches)
        """
        H = self.homographies[index]
        data1, data2 = self.ref_imgs[index], self.warped_imgs[index]
        if self.transform is not None:
            data1 = self.transform(data1)
            data2 = self.transform(data2)
        return data1, data2, H

    def __len__(self):
        return len(self.homographies)

    def _check_datafile_exists(self):
        return os.path.exists(self.data_file)

    def _check_downloaded(self):
        return os.path.exists(self.data_dir)

    def download(self):
        if self._check_datafile_exists():
            print('# Found cached data {}'.format(self.data_file))
            return

        if not self._check_downloaded():
            for k, v in self.urls.iteritems():
                # download files
                url = self.urls[k][0]
                filename = self.urls[k][1]
                md5 = self.urls[k][2]
                fpath = os.path.join(self.root, filename)

                download_url(url, self.root, filename, md5)

                print('# Extracting data {}\n'.format(self.data_down))

                import tarfile
                if (fpath.endswith("tar.gz")):
                    with tarfile.open(fpath, "r:gz")as z:
                        z.extractall(self.data_dir)
                os.unlink(fpath)

        # process and save as torch files
        print('# Caching data {}'.format(self.data_file))
        import json
        from pprint import pprint
        #print self.urls['splits']
        with open(os.path.join(self.root, self.urls['splits'][1])) as splits_file:    
            data = json.load(splits_file)
        if self.train:
            self.img_fnames = data[self.split_name]['train']
        else:
            self.img_fnames = data[self.split_name]['test']
        #pprint(data)
        dataset =  read_images_and_homographies(self.data_dir, self.image_ext, self.img_fnames);
        with open(self.data_file, 'wb') as f:
            torch.save(dataset, f)


def read_images_and_homographies(data_dir, image_ext, valid_list):
    """Return a Tensor containing the patches
    """
    def PIL2array(_img):
        """Convert PIL image type to numpy 2D array
        """
        return np.array(_img.getdata(), dtype=np.uint8).squeeze()

    ref_images = []
    warped_images = []
    homographies = []
    for file_dir in os.listdir(data_dir):
        print(file_dir)
        if file_dir not in valid_list:
            continue
        current_files = []
        for fname  in os.listdir(os.path.join(data_dir, file_dir)):
            print(fname)
            if fname.endswith(image_ext):
                current_files.append(os.path.join(os.path.join(data_dir, file_dir), fname))
            sorted(current_files)
        ref_img  =  np.array(Image.open(current_files[0])).mean(axis = 2)
        h,w = ref_img.shape
        ref_img = ref_img.reshape((1,1,h,w))
        for i in range(1, len(current_files)):
            curr_fname = current_files[i]
            warped_img =  np.array(Image.open(current_files[i])).mean(axis = 2)
            h1,w1 = warped_img.shape
            warped_img = warped_img.reshape((1,1,h1,w1))
            h_fname = os.path.join(os.path.join(data_dir, file_dir), 'H_1_' + str(i+1))
            H = torch.FloatTensor(np.loadtxt(h_fname))
            ref_images.append(torch.ByteTensor(ref_img))
            warped_images.append(torch.ByteTensor(warped_img))
            homographies.append(H)
    return (ref_images, warped_images, homographies)
