# from mmdet.datasets import PIPELINES
from mmdet.registry import TRANSFORMS
from utils import load_image, norma
import os.path as osp


@TRANSFORMS.register_module()
class LoadRSImageFromFile:
    def __init__(self,
                 bands_index):
        self.bands_index = bands_index

    def __call__(self, results):
        filename = results['img_path']
        # if results['img_prefix'] is not None:
            # filename = osp.join(results['img_prefix'], results['img_path'])
        # else:
            # filename = results['img_path']

        img = load_image(filename)
        img = img[:, :, self.bands_index]

        # results['filename'] = filename
        # results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['img_fields'] = ['img']

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'bands_index={self.bands_index}')
        return repr_str


@TRANSFORMS.register_module()
class Norma:
    def __init__(self, method):
        self.method = method

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            results[key] = norma(results[key], self.method)
        results['norma_method'] = self.method
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'normalization method used: {self.method}'
        return repr_str

# @PIPELINES.register_module()
