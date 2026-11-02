from pathlib import Path
import numpy as np
import SimpleITK as sitk
TARGET_SPACING_XYZ=(2.0,2.0,3.0)
PATCH_SIZE_XYZ=(80,112,64)
EXPECTED_ARRAY_SHAPE_ZYX=(64,112,80)
SPACING_ATOL_MM=1e-5
ORIGIN_ATOL_MM=0.001
DIRECTION_ATOL=1e-5
CT_CLIP_MIN_HU=-1000.0
CT_CLIP_MAX_HU=2000.0
DOSE_NORMALIZATION_GY=70.0
DOSE_NORMALIZED_MAX=1.2

def read_3d_image(path: Path) -> sitk.Image:
    """读取三维医学影像。"""
    if path is None:
        raise FileNotFoundError('影像路径为空')
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'找不到文件：{path}')
    if not path.is_file():
        raise FileNotFoundError(f'路径不是文件：{path}')
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError(f'不是三维影像：{path}；dimension={image.GetDimension()}')
    return image

def same_geometry(reference: sitk.Image, image: sitk.Image) -> bool:
    """比较size、spacing、origin、direction。"""
    return tuple(reference.GetSize()) == tuple(image.GetSize()) and np.allclose(reference.GetSpacing(), image.GetSpacing(), atol=SPACING_ATOL_MM, rtol=0) and np.allclose(reference.GetOrigin(), image.GetOrigin(), atol=ORIGIN_ATOL_MM, rtol=0) and np.allclose(reference.GetDirection(), image.GetDirection(), atol=DIRECTION_ATOL, rtol=0)

def binary_array(image: sitk.Image) -> np.ndarray:
    """mask转换为0/1数组，顺序Z,Y,X。"""
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    return (array > 0.5).astype(np.uint8)

def mask_bbox_center_physical(mask_image: sitk.Image):
    """
    根据mask三维包围盒计算中心。

    返回：
    1. center_physical_xyz
    2. center_continuous_index_xyz
    """
    array = binary_array(mask_image)
    coordinates_zyx = np.argwhere(array > 0)
    if coordinates_zyx.size == 0:
        raise ValueError('oral mask为空，无法确定裁剪中心')
    minimum_zyx = coordinates_zyx.min(axis=0).astype(float)
    maximum_zyx = coordinates_zyx.max(axis=0).astype(float)
    center_zyx = (minimum_zyx + maximum_zyx) / 2.0
    center_xyz = (float(center_zyx[2]), float(center_zyx[1]), float(center_zyx[0]))
    center_physical_xyz = mask_image.TransformContinuousIndexToPhysicalPoint(center_xyz)
    return (tuple((float(value) for value in center_physical_xyz)), tuple((float(value) for value in center_xyz)))

def create_patch_reference(ct_image: sitk.Image, center_physical_xyz) -> sitk.Image:
    """
    创建固定patch参考网格。

    patch方向继承CT；
    patch中心与oral包围盒物理中心一致。
    """
    direction_matrix = np.asarray(ct_image.GetDirection(), dtype=float).reshape(3, 3)
    spacing = np.asarray(TARGET_SPACING_XYZ, dtype=float)
    size = np.asarray(PATCH_SIZE_XYZ, dtype=float)
    center_index = (size - 1.0) / 2.0
    center_physical = np.asarray(center_physical_xyz, dtype=float)
    origin = center_physical - direction_matrix @ (center_index * spacing)
    reference = sitk.Image([int(value) for value in PATCH_SIZE_XYZ], sitk.sitkFloat32)
    reference.SetSpacing(tuple((float(value) for value in TARGET_SPACING_XYZ)))
    reference.SetDirection(ct_image.GetDirection())
    reference.SetOrigin(tuple((float(value) for value in origin)))
    return reference

def resample_to_reference(image: sitk.Image, reference: sitk.Image, interpolator, default_value, output_pixel_type) -> sitk.Image:
    """重采样到固定patch空间。"""
    return sitk.Resample(image, reference, sitk.Transform(), interpolator, float(default_value), output_pixel_type)

def patch_retained_fraction(original_mask: sitk.Image, patch_reference: sitk.Image) -> float:
    """
    计算原始ROI落入最终patch的比例。

    将patch视野反向映射到原始mask空间，
    避免因目标spacing改变造成体素比例误差。
    """
    patch_ones_array = np.ones(EXPECTED_ARRAY_SHAPE_ZYX, dtype=np.uint8)
    patch_ones_image = sitk.GetImageFromArray(patch_ones_array)
    patch_ones_image.CopyInformation(patch_reference)
    patch_fov_on_original = sitk.Resample(patch_ones_image, original_mask, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    original_array = binary_array(original_mask) > 0
    fov_array = sitk.GetArrayFromImage(patch_fov_on_original) > 0
    denominator = int(original_array.sum())
    if denominator == 0:
        return np.nan
    numerator = int(np.logical_and(original_array, fov_array).sum())
    return float(numerator / denominator)

def physical_volume_cc(mask_image: sitk.Image) -> float:
    """计算mask物理体积，单位cc。"""
    array = binary_array(mask_image)
    voxel_volume_mm3 = float(np.prod(mask_image.GetSpacing()))
    return float(array.sum() * voxel_volume_mm3 / 1000.0)

def prepare_arrays(ct_patch: sitk.Image, dose_patch: sitk.Image, oral_patch: sitk.Image, gtv_patch: sitk.Image, dose_scale_to_gy: float):
    """生成最终NPZ中的四个影像数组。"""
    ct_raw = sitk.GetArrayFromImage(ct_patch).astype(np.float32)
    dose_raw = sitk.GetArrayFromImage(dose_patch).astype(np.float32)
    oral = binary_array(oral_patch)
    gtv = binary_array(gtv_patch)
    ct_raw = np.nan_to_num(ct_raw, nan=CT_CLIP_MIN_HU, posinf=CT_CLIP_MAX_HU, neginf=CT_CLIP_MIN_HU)
    dose_raw = np.nan_to_num(dose_raw, nan=0.0, posinf=0.0, neginf=0.0)
    ct_clipped = np.clip(ct_raw, CT_CLIP_MIN_HU, CT_CLIP_MAX_HU)
    ct_normalized = (2.0 * (ct_clipped - CT_CLIP_MIN_HU) / (CT_CLIP_MAX_HU - CT_CLIP_MIN_HU) - 1.0).astype(np.float32)
    dose_gy = dose_raw * float(dose_scale_to_gy)
    dose_gy = np.clip(dose_gy, 0.0, DOSE_NORMALIZATION_GY * DOSE_NORMALIZED_MAX)
    dose_normalized = np.clip(dose_gy / DOSE_NORMALIZATION_GY, 0.0, DOSE_NORMALIZED_MAX).astype(np.float32)
    arrays = {'ct': ct_normalized.astype(np.float32), 'dose': dose_normalized.astype(np.float32), 'oral': oral.astype(np.uint8), 'gtv': gtv.astype(np.uint8), 'ct_hu_for_qc': ct_clipped.astype(np.float32), 'dose_gy_for_qc': dose_gy.astype(np.float32)}
    for name in ['ct', 'dose', 'oral', 'gtv']:
        if arrays[name].shape != EXPECTED_ARRAY_SHAPE_ZYX:
            raise RuntimeError(f'{name}输出形状错误：{arrays[name].shape}；预期={EXPECTED_ARRAY_SHAPE_ZYX}')
    if not np.isfinite(arrays['ct']).all():
        raise RuntimeError('CT归一化后存在NaN或Inf')
    if not np.isfinite(arrays['dose']).all():
        raise RuntimeError('dose归一化后存在NaN或Inf')
    if int(arrays['oral'].sum()) <= 0:
        raise RuntimeError('输出oral mask为空')
    if int(arrays['gtv'].sum()) <= 0:
        raise RuntimeError('输出GTV mask为空')
    if float(arrays['ct'].min()) < -1.0001 or float(arrays['ct'].max()) > 1.0001:
        raise RuntimeError('CT归一化范围异常')
    if float(arrays['dose'].min()) < -1e-06 or float(arrays['dose'].max()) > DOSE_NORMALIZED_MAX + 1e-06:
        raise RuntimeError('dose归一化范围异常')
    return arrays