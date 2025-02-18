import rioxarray
import torch
import numpy as np
from time import time
import argparse
from mmdet.apis import init_detector, inference_detector
# from nms_area import nms, detections_div
from utils import bboxes_to_sqlite, rtree_nms, detections_div, detections_to_bboxes
from shapely.geometry import Polygon, box


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', dest='dir', type=str,
                        help='directory', default='dino_swim')

    parser.add_argument('-c', '--cfg', dest='cfg', type=str,
                        help='config file path', default='dino_swim_config')

    parser.add_argument('-ck', '--checkpoint', dest='checkpoint', type=str,
                        help='checkpoint file from save model', default='epoch_30')

    parser.add_argument('-chunk', '--chunk_size', dest='chunk_size', type=int,
                        help='chunk size for tile processing', default=5000)

    parser.add_argument('-s', '--stride', dest='stride', type=int,
                        help='stride to create overlapping image patches for inference',
                        default=160)

    parser.add_argument('-o', '--overlap', dest='overlap', type=int,
                        help='stride to create overlapping image tiles for inference',
                        default=160)
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int,
                        help='batch size for model prediction',
                        default=8)

    parser.add_argument('-iou', '--iou_threshold', dest='iou_threshold', type=float,
                        help='threshold to remove duplicate bboxes',
                        default=0.8)

    parser.add_argument('-j', '--jahr', dest='jahr', type=str, help='2016 and 2020', default='2016')

    parser.add_argument('-nms', '--nms', dest='nms', type=int, help='apply nms to get rid of partial boxes or not',
                        default=1)
    args = parser.parse_args()
    return args


def get_chunk_trafo(arr):
    xr = arr.coords['x'].data
    yr = arr.coords['y'].data

    xres, yres = arr.rio.transform()[0], arr.rio.transform()[4]
    xskew, yskew = arr.rio.transform()[1], arr.rio.transform()[3]
    return xres, xskew, min(xr), yskew, yres, max(yr)


if __name__ == '__main__':
    args = args()
    jahr = args.jahr
    # some file path and parameters
    stride = args.stride
    overlap = args.overlap
    iou_threshold = args.iou_threshold
    # define chunk size
    chunk_size = args.chunk_size
    batch_size = args.batch_size

    config = f'log/{args.dir}/{args.cfg}.py'
    checkpoint = f'log/{args.dir}/{args.checkpoint}.pth'
    files = dict({"2016": 'WV3_2016-11_north.tif',
                  "2020": 'WV3_2020-12_north.tif',
                  "SH": 'SH_16APR180_cubics.tif',
                  "DL": 'DL_17SEP28_cubics.tif'})
    large_file_path = f'/data_hdd/bangalore/raster/{files[jahr]}'
    print(f'#######Predicting on {large_file_path}#######')
    # large_file_path = f'/data_hdd/bangalore/raster/SH_16APR180.tif'
    # large_file_path = f'/data_hdd/bangalore/raster/DL_17SEP28.tif'
    # load trained model
    model = init_detector(config=config,
                          checkpoint=checkpoint,
                          device='cuda')

    # use rioxarray to load large image file
    array = rioxarray.open_rasterio(large_file_path)
    # get crs info
    crs_ = array.rio.crs
    # shape info
    nbands, height, width = array.shape
    # print(array.shape)

    # how many chunks in large image
    nchunks_h = len(range(0, height, chunk_size-overlap))
    nchunks_w = len(range(0, width, chunk_size-overlap))
    nchunks = nchunks_h * nchunks_w
    print("Chunk size for processing: {}".format(chunk_size))
    N_trees = []
    True_bboxes_b,  True_bboxes_c = [], []
    t0 = time()
    # height, width
    for i, y in enumerate(range(0, height, chunk_size-overlap)):
        for j, x in enumerate(range(0, width, chunk_size-overlap)):
            idx = i * nchunks_w + j + 1
            print(f"Loading chunk {idx}/{nchunks}")
            t1 = time()
            chunk = array[:, y: y + chunk_size, x: x + chunk_size].load()
            # chunk transform info
            xres, xskew, xr_min, yskew, yres, yr_max = get_chunk_trafo(chunk)
            img = chunk.data
            if img.shape[0] > 3:
                img = img[[5, 3, 1], :, :].transpose((1, 2, 0))
            else:
                img = img.transpose((1, 2, 0))

            t2 = time()
            disk_load_time = t2 - t1
            print("chunk data prediction....")
            # make prediction on chunk data to output detections
            x_tiles = int(np.ceil(img.shape[1] / stride))
            y_tiles = int(np.ceil(img.shape[0] / stride))

            end2 = (x_tiles + 1) * stride - 320
            end1 = (y_tiles + 1) * stride - 320
            y_range = range(0, end1 if end1 > 1 else 1, stride)
            x_range = range(0, end2 if end2 > 1 else 1, stride)

            y_pad = y_range[-1] + 320 - img.shape[0]
            x_pad = x_range[-1] + 320 - img.shape[1]

            img = np.pad(img, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant', constant_values=(0, 0))

            n_patches = len(y_range) * len(x_range)
            print(f'{n_patches} patches needed to be predicted in this Chunk')

            def patch_generator():
                for y1 in y_range:
                    for x1 in x_range:
                        yield img[y1: y1 + 320, x1: x1 + 320, :]

            patch_gen = patch_generator()

            patch_idx = 0
            y_, x_ = 0, 0
            Bboxes, Scores, Ul = [], [], []

            while patch_idx < n_patches:
                pre_batch_size = min(batch_size, n_patches, n_patches - patch_idx)
                patch_idx += pre_batch_size
                batch, ul = [], []
                for j in range(pre_batch_size):
                    # batch[j] = next(patch_gen)
                    batch.append(next(patch_gen))
                    ul.append([x_, y_, x_, y_])

                    x_ += stride
                    if x_ + 320 > img.shape[1]:
                        x_ = 0
                        y_ += stride

                # make predictions on batch using model
                detections = inference_detector(model, batch)
                bboxes = [detection.pred_instances['bboxes'] for detection in detections]
                scores = [detection.pred_instances['scores'] for detection in detections]
                # masks = [detection.pred_instances['masks'] for detection in detections]

                Bboxes += bboxes
                Scores += scores
                # Masks += masks
                Ul += ul

            # processing detection
            New_bboxes, New_ul = [], []
            for bb, ul in zip(Bboxes, Ul):
                ul_new = np.tile(np.array(ul), len(bb)).reshape(len(bb), 4)
                ul_new_tensor = torch.from_numpy(ul_new).to('cuda')
                new_bb = bb + ul_new_tensor

                New_bboxes.append(new_bb)
                New_ul.append(ul_new_tensor)

            New_bboxes_merge = torch.concat(New_bboxes, dim=0)
            Scores_merge = torch.concat(Scores, dim=0)
            # Masks_merge = torch.concat(Masks, dim=0)
            Ul_merge = torch.concat(New_ul, dim=0)

            detections = torch.concat((New_bboxes_merge, Scores_merge.reshape((-1, 1))), dim=-1).cpu()

            # filter with classification score with 0.5
            detections = detections[detections[:, -1] > 0.4]
            print(f'chunk data prediction done and found {detections.shape[0]}')
            # apply nms to remove partial and duplicate bboxes
            detections = detections_to_bboxes(detections)
            if args.nms == 1:
                dets = rtree_nms(detections)
            else:
                dets = detections
            t3 = time()
            inf_time = t3 - t2
            N_trees.append(len(dets))
            print(f'Chunk loading time: {int(disk_load_time)}s, inference + Patches NMS: {int(inf_time)}s')
            print(f'After Patches NMS {len(dets)} trees')
            print(f'{np.sum(N_trees)} have been found cumulatively')
            # no nms
            dets_b, dets_c = detections_div(chunk_size, overlap, dets)
            # bbox output to sqlit
            true_bboxes_b = [[b.bounds[0] * xres + xr_min,
                              b.bounds[1] * yres + yr_max,
                              b.bounds[2] * xres + xr_min,
                              b.bounds[3] * yres + yr_max] for b in dets_b]
            true_bboxes_c = [[b.bounds[0] * xres + xr_min,
                              b.bounds[1] * yres + yr_max,
                              b.bounds[2] * xres + xr_min,
                              b.bounds[3] * yres + yr_max] for b in dets_c]
            True_bboxes_b += true_bboxes_b
            True_bboxes_c += true_bboxes_c

            print(f'Detections at overlapping area in this chunk: {len(dets_b)}, '
                  f'in all detected area: {len(True_bboxes_b)}',
                  f'Detections at center area in this chunk: {len(dets_c)}, '
                  f'in all detected area: {len(True_bboxes_c)}')
            print('###############################################')
            print(' ')

    True_bboxes_b = [box(b[0], b[1], b[2], b[3]) for b in True_bboxes_b]
    True_bboxes_c = [box(b[0], b[1], b[2], b[3]) for b in True_bboxes_c]

    bboxes_to_sqlite(bboxes=True_bboxes_b,
                     dst_name=f'bboxes/{args.dir}_s_{args.stride}_o_{args.overlap}_bb_b_{jahr}_nms_0.sqlite',
                     crs=crs_)
    bboxes_to_sqlite(bboxes=True_bboxes_c,
                     dst_name=f'bboxes/{args.dir}_s_{args.stride}_o_{args.overlap}_bb_c_{jahr}_nms_{args.nms}.sqlite',
                     crs=crs_)
    if args.nms == 1:
        True_bboxes_b = rtree_nms(True_bboxes_b)
        bboxes_to_sqlite(bboxes=True_bboxes_b,
                         dst_name=f'bboxes/{args.dir}_s_{args.stride}_o_{args.overlap}_bb_b_{jahr}_nms_{args.nms}.sqlite',
                         crs=crs_)
        True_bboxes_b = True_bboxes_b.tolist()

    print(len(True_bboxes_b), len(True_bboxes_c))

    True_bboxes = True_bboxes_b + True_bboxes_c
    bboxes_to_sqlite(bboxes=True_bboxes,
                     dst_name=f'bboxes/{args.dir}_s_{args.stride}_o_{args.overlap}_bb_{jahr}_whole_{args.nms}.sqlite',
                     crs=crs_)
    print('bboxes output to sqlit success')
    # bboxes_to_sqlit(Polygons,
    #                 f'{args.dir}_{args.iou_threshold}_{args.stride}_polygons.sqlite', crs=crs_)
    # print('polygons output to sqlit success')

    print(f'Total processing time: {(int(time()) - t0)}s')
    print(f'In the whole image, {len(True_bboxes)} trees have been found!!!!')
