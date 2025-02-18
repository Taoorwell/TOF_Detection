from osgeo import gdal
from shapely import box, Polygon
from shapely.strtree import STRtree
import numpy as np
from osgeo import ogr
import fiona
from shapely.geometry.geo import mapping
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap



def load_image(file_path):
    ds = gdal.Open(file_path)
    image = np.empty((ds.RasterYSize, ds.RasterXSize, ds.RasterCount), dtype=np.float32)
    for b in range(ds.RasterCount):
        array_band = ds.GetRasterBand(b+1).ReadAsArray()
        image[:, :, b] = array_band
    # add ndvi
    ndvi = (image[:, :, 5] - image[:, :, 3])/(image[:, :, 5] + image[:, :, 3] + 1E-10)
    ndvi = ndvi[:, :, np.newaxis]
    image = np.concatenate((image, ndvi), axis=-1)
    return image


def norma(image, method='min-max'):
    mi_1 = np.percentile(image, 1, axis=(0, 1)).reshape((1, 1, -1))
    ma_99 = np.percentile(image, 99, axis=(0, 1)).reshape((1, 1, -1))
    mi_30 = np.percentile(image, 30, axis=(0, 1)).reshape((1, 1, -1))
    ma_70 = np.percentile(image, 70, axis=(0, 1)).reshape((1, 1, -1))

    if method == 'dw':
        image = np.log(image * 0.0001 + 1)
        image = (image - mi_30 * 0.0001) / (ma_70 * 0.0001)
        image = np.exp(image * 5 - 1)
        image = image / (image + 1)
    else:
        image = ((image - mi_1) / (ma_99 - mi_1)).clip(0, 1)
    image = image.astype(np.float32)
    return image


def overlap(a: Polygon, b: Polygon) -> float:
    intersection = a.intersection(b)
    return intersection.area / b.area
def filter_bboxes(bboxes, tmin=0.5, tmax=2):
    ratio = [(box.bounds[2] - box.bounds[0]) / (box.bounds[3] - box.bounds[1]) for box in bboxes]
    mask = (np.array(ratio) > 0.5) & (np.array(ratio) < 2)
    bboxes = bboxes[mask]
    return bboxes

def rtree_nms(bboxes, iou_threshold=0.8):
    """
    List of shapely polygons
    """
    areas = [box.area for box in bboxes]
    sorted_indices = np.argsort(areas)
    bboxes = np.array(bboxes)[sorted_indices[::-1]]
    # mask out 
    bboxes = filter_bboxes(bboxes)

    box_mask = np.ones(len(bboxes))

    tree = STRtree(bboxes)

    result_polygons = []

    for index, (current_bbox, mask_val) in enumerate(zip(bboxes, box_mask)):
        if mask_val == 0:
            continue  # skip this polygon

        result_polygons.append(current_bbox)

        # get polygons that are close by
        adjacent_indices = tree.query(current_bbox)

        for i in adjacent_indices:
            adjacent_box = tree.geometries.take(i)

            # calculate iou with current bbox
            if overlap(current_bbox, adjacent_box) > iou_threshold and current_bbox != adjacent_box:
                # if the iou with the current box is too high, we throw the polygon out
                box_mask[i] = 0

        # if index % 1000 == 0:
            # print(index)

    return np.array(bboxes)[box_mask == 1]


def load_bboxes_from_file(fname):
    shapefile = ogr.Open(fname)
    layer = shapefile.GetLayer()
    boxes = []
    for i, feature in enumerate(layer):
        geometry = feature.GetGeometryRef()
        minx, maxx, miny, maxy = geometry.GetEnvelope()
        boxes.append(box(minx, miny, maxx, maxy))
    return boxes


def detections_to_bboxes(detections):
    detections = detections.tolist()
    boxes = [box(d[0], d[1], d[2], d[3]) for d in detections]
    return boxes


def detections_div(chunk_size, over, detections):
    center_box = box(over, over, chunk_size-over, chunk_size-over)
    detections_boundary, detections_center = [], []
    for det in detections:
        iom = overlap(center_box, det)
        if iom == 1:
            detections_center.append(det)
        else:
            detections_boundary.append(det)
    return detections_boundary, detections_center


def bboxes_to_sqlite(bboxes, dst_name, crs):
    schema = {"geometry": "Polygon",
              "properties": {"id": "int"}}
    records = [{"geometry": mapping(p), "properties": {"id": i}} for i, p in enumerate(bboxes)]
    with fiona.open(dst_name, mode='w', crs=crs, driver="SQLite", schema=schema) as f:
        f.writerecords(records)


def plot_masks(masks):
    for mask in masks:
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        ma = mask * 1
        ma = np.ma.masked_where(ma == 0, ma)
        cmap = ListedColormap([c, [0, 0, 0]])
        plt.imshow(ma, cmap=cmap, alpha=0.7)
