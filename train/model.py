"""3D U-Net (MONAI DynUNet, nnU-Net style). Filters are smaller than nnU-Net's default so a Kaggle T4 keeps up."""
from monai.networks.nets import DynUNet

FILTERS = (32, 64, 128, 256, 320, 320)


def build_model(num_classes, filters=FILTERS):
    n = len(filters)
    return DynUNet(
        spatial_dims=3, in_channels=1, out_channels=num_classes,
        kernel_size=[3] * n, strides=[1] + [2] * (n - 1), upsample_kernel_size=[2] * (n - 1),
        filters=list(filters), norm_name="instance", deep_supervision=False, res_block=True,
    )
