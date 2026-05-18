import os

from cloud_filter import extract_cloud
from IOU_calculate import calculate_cloud_iou

inpath = r"C:/Users/kyudo/Desktop/DIP_project/main/"
rgb_image = "gk2a_ami_le1b_rgb-s-true_ko020lc_202605130630.png"
nc_file = "gk2a_ami_le2_cld_ko020lc_202605131530.nc"

rgb_path = os.path.join(
    inpath,
    rgb_image
)
nc_path = os.path.join(
    inpath,
    nc_file
)



mask, result = extract_cloud(rgb_path)

iou = calculate_cloud_iou(nc_path,mask)

print("IoU = ",iou)

#할 것 : 이미지 크기 맞추기
