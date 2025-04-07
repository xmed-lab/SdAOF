import math
import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence, Union, Tuple, List, Dict, Optional
from pathlib import Path
from omegaconf import DictConfig, ListConfig, OmegaConf
from matplotlib.ticker import FuncFormatter
from math import floor, log10
ConfigType = Union[DictConfig, ListConfig]

def add_tuple(a, b):
    return tuple(a_i + b_i for a_i, b_i in zip(a, b))


def subtract_tuple(a, b):
    return tuple(a_i - b_i for a_i, b_i in zip(a, b))


def divide_tuple_scalar(a, b):
    b = (b,) * len(a)
    return tuple(a_i / b_i for a_i, b_i in zip(a, b))


def divide_tuple(a, b):
    return tuple(a_i / b_i for a_i, b_i in zip(a, b))


def multiply_tuple(a, b):
    return tuple(a_i * b_i for a_i, b_i in zip(a, b))


def ceil_tuple(a):
    return tuple(math.ceil(a_i) for a_i in a)


def floor_tuple(a):
    return tuple(math.floor(a_i) for a_i in a)

def all_elements_equal(arr:List):
    '''
    >>> all_elements_equal([1.0,1.0,1.0])
    True
    >>> all_elements_equal([1.0,1.5,1.0])
    False
    '''
    return arr.count(arr[0]) == len(arr)

def human_readable_formatter(value, pos):
    """
    make matplotlib tickmarks with large number more readable
    e.g. 100000 -> 1M
    https://stackoverflow.com/questions/61330427/set-y-axis-in-millions
    """
    num_thousands = 0 if abs(value) < 1000 else floor(log10(abs(value)) / 3)
    value = round(value / 1000**num_thousands, 2)
    return f"{value:g}" + " KMGTPEZY"[num_thousands]

def scatterplot_1d(data: List, label, color="#E24A33", fig=None, ax=None):
    """scatterplot and return updated figure"""
    if ax is None:
        fig, ax = plt.subplots()
    plt.scatter(x=range(len(data)), y=data, label=label, s=6, c=color)
    ax.yaxis.set_major_formatter(FuncFormatter(human_readable_formatter))
    return fig, ax


def horizontal_line(y_intercept, label, color, fig=None, ax=None):
    """plot a horizontal line and return the updated plot"""
    if ax is None:
        fig, ax = plt.subplots()
    plt.axhline(y=y_intercept, linestyle="--", label=label, color=color)
    return fig, ax

def write_csv(data: Dict[str, List], column_names: Optional[List[str]], file_path):
    """
    write data into csv

    The dictionary should consist of key:List as key value pairs representing a single row.
    Optional name of each columns may also be provided
    """
    df = pd.DataFrame.from_dict(data, orient="index", columns=column_names)
    df.to_csv(file_path)

def get_segmentation_stats(
    segmentation: sitk.Image,
) -> sitk.LabelShapeStatisticsImageFilter:
    """return sitk filter obj containing segmentation stats"""
    fltr = sitk.LabelShapeStatisticsImageFilter()
    fltr.Execute(sitk.Cast(segmentation, sitk.sitkUInt8))
    return fltr

def get_nifti_stem(path) -> str:
    """
    '/home/user/image.nii.gz' -> 'image'
    1.3.6.1.4.1.14519.5.2.1.6279.6001.905371958588660410240398317235.nii.gz ->1.3.6.1.4.1.14519.5.2.1.6279.6001.905371958588660410240398317235
    """

    def _get_stem(path_string) -> str:
        name_subparts = Path(path_string).name.split(".")
        return ".".join(name_subparts[:-2])  # get rid of [nii, gz]

    return _get_stem(path)

def load_components(config: ConfigType, basepath, special_key) -> ConfigType:
    """
    update dict if the key == special_key
    return a updated dict
    """
    if config is not None and special_key in config:
        loaded_config = OmegaConf.load(basepath / config.pop(special_key))
        updated_config = OmegaConf.merge(loaded_config, config)
        return updated_config
    else:
        return config

def combine_segmentations(imgs: List[sitk.Image], ref_img: sitk.Image, fill_label=1):
    """Combine multiple segmentation images into a single segmentation image.

    Precondition:
        segmentation masks do not overlap

    Postcondition:
        a new segmentation images is returned where voxels are filled with fill_label
        if the voxel position is labelled in one of the segmentation image
    """
    new_seg = np.zeros_like(sitk.GetArrayFromImage(ref_img))

    for seg in imgs:
        single_seg = sitk.GetArrayViewFromImage(seg)
        new_seg[single_seg > 0.5] = fill_label

    img_out = sitk.GetImageFromArray(new_seg)
    img_out.CopyInformation(ref_img)
    return img_out

def write_image(img, out_path, pixeltype=None):
    """save image"""
    if isinstance(out_path, Path):
        out_path = str(out_path)
    if pixeltype:
        img = sitk.Cast(img, pixeltype)
    sitk.WriteImage(img, out_path)

def reorient_to(img, axcodes_to: Union[sitk.Image, str] = "PIR", verb=False):
    """Reorients the Image from its original orientation to another specified orientation

    adapted from https://github.com/anjany/verse/blob/main/utils/data_utilities.py

    Parameters:
    ----------
    img: SimpleITK image
    axcodes_to: a string of 3 characters specifying the desired orientation

    Returns:
    ----------
    new_img: The reoriented SimpleITK image

    """
    if isinstance(axcodes_to, sitk.Image):
        axcodes_to = get_orientation_code_itk(axcodes_to)

    new_img = sitk.DICOMOrient(img, axcodes_to)

    if verb:
        print(
            "[*] Image reoriented from", get_orientation_code_itk(img), "to", axcodes_to
        )
    return new_img

def read_image(img_path) -> sitk.Image:
    """returns the SimpleITK image read from given path

    Parameters:
    -----------
    pixeltype (ImagePixelType):
    """
    img_path = Path(img_path).resolve()
    img_path = str(img_path)

    return sitk.ReadImage(img_path)

def read_config_and_load_components(filepath, special_key="_load"):
    """read yaml from filepath and load subcomponents"""
    config_dict = OmegaConf.load(filepath)
    assert isinstance(config_dict, DictConfig)
    for key in config_dict:
        config_dict[key] = load_components(
            config_dict[key], Path(filepath).parent, special_key
        )
    return config_dict

def get_segmentation_labels(segmentation: sitk.Image):
    """return segmentation labels"""
    fltr = sitk.LabelShapeStatisticsImageFilter()
    fltr.Execute(sitk.Cast(segmentation, sitk.sitkUInt8))
    return fltr.GetLabels()

def get_orientation_code_itk(img_or_affine_mtrx: Union[sitk.Image, Sequence]) -> str:
    """Orientation is a tricky topic:
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Orientation%20Explained

    """
    if isinstance(img_or_affine_mtrx, sitk.Image):
        return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            img_or_affine_mtrx.GetDirection()
        )
    else:
        return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            img_or_affine_mtrx
        )

def physical_size_to_voxel_size(img, physical_size) -> Tuple:
    """Given an img, how many voxels(rounded evenly) does it require to represent a given physical size?"""
    return tuple(
        [int(np.around(p / sp)) for p, sp in zip(physical_size, img.GetSpacing())]
    )

def infer_roi_origin_from_center(centre, roi_size):
    """Given a ROI of size `roi_size` in voxel units and whose centroid index is `centre`,
    return the corresponding starting index of the ROI in the image
    +-------------------------+  |
    |t(tx,ty,tz)              |  |
    |                         |  |
    |                         |  |
    |          c              |  sy
    |                         |  |
    |                         |  |
    |                         |  |
    +-------------------------+  |
    ------------sx-------------
    """

    t = tuple([int(c - s // 2) for c, s in zip(centre, roi_size)])
    return t

def required_padding(volume, volume_size, centroid_index, verbose=True):
    """how much padding is required to be able to extract ROI of voxel_size whose centroid index is at centroid_index"""
    upperbound_index: tuple = add_tuple(
        centroid_index, divide_tuple_scalar(volume_size, 2)
    )
    lowerbound_index: tuple = subtract_tuple(
        centroid_index, divide_tuple_scalar(volume_size, 2)
    )

    upperbound_pad = tuple(
        [
            int(max(0, ub_idx - im_idx))
            for im_idx, ub_idx in zip(volume.GetSize(), upperbound_index)
        ]
    )
    lowerbound_pad = tuple([int(max(0, -lb_idx)) for lb_idx in lowerbound_index])

    if verbose:
        print(
            f"target voxel {volume_size} lowerbound {lowerbound_pad} upperbound {upperbound_pad}"
        )
    np.testing.assert_array_equal(
        np.array(volume_size),
        np.array(subtract_tuple(upperbound_index, lowerbound_index)),
    )

    return lowerbound_pad, upperbound_pad

def extract_bbox(img, seg, label_id, physical_size, padding_value, verbose=True):
    """extract ROI from img of given physical size by finding the bounding box with label id from seg image

    Args:
        img (sitk.Image):
        seg (sitk.Image):
        physical_size (tuple): _description_
        padding_value (scalar): value to fill in for region outside of the image
        verbose (bool, optional): print additional information. Defaults to True.
    """
    assert isinstance(img, sitk.Image)
    assert isinstance(seg, sitk.Image)

    voxel_size = physical_size_to_voxel_size(img, physical_size)

    # execute filter to obtain bounding box and centroid of given segmentation label
    fltr = sitk.LabelShapeStatisticsImageFilter()
    fltr.Execute(seg)

    labels = fltr.GetLabels()

    # make sure the label being asked for exists in the segmentation map
    assert label_id in labels

    # pad around Bounding Box centroid to attain given voxel size
    centroid_index = img.TransformPhysicalPointToIndex(fltr.GetCentroid(label_id))
    lb, ub = required_padding(img, voxel_size, centroid_index, verbose=verbose)
    lb_padded = add_tuple(lb, (50,) * 3)
    ub_padded = add_tuple(
        ub, (50,) * 3
    )  # the exact padding can be off due to floating point ops, hence add safety padding
    padded_img: sitk.Image = sitk.ConstantPad(img, lb_padded, ub_padded, padding_value)

    # find the index of the centroid in the padded image
    padded_centroid_index = padded_img.TransformPhysicalPointToIndex(
        fltr.GetCentroid(label_id)
    )

    # get the start of the ROI
    roi_start_index = infer_roi_origin_from_center(padded_centroid_index, voxel_size)
    region_of_interest: sitk.Image = sitk.RegionOfInterest(
        padded_img, voxel_size, roi_start_index
    )

    if verbose:
        print(f"Label Bounding Box: {fltr.GetBoundingBox(label_id)}")
        print(
            f"Coordinates of Segmentation Centroid {img.TransformPhysicalPointToIndex(fltr.GetCentroid(label_id))}"
        )

    return region_of_interest

def get_stem(path):
    """wrap Path.stem"""
    return Path(path).stem