import os
rgb_mean_v = [111.97261949846957, 118.42392668739762, 115.031314378774]
rgb_std_v = [72.97088063, 72.98526569, 81.91774359]
sar_mean_v = [63.30051921735858]
sar_std_v = [68.20405016]

root_path = "/data/yhzhou23/data/EarthM3/"

# 城市列表
train_cities = [
    "America-Eugene",
    "America-Louisville",
    "French-Paris",
    "Morocco-Casablanca",
    "Nanjing",
    "Netherlands-Rotterdam",
    "Singapore",
]

val_cities = [
    "Australia-PortHedland",
    "America-Pake",
    "Russian-Engels",
]

test_cities = [
    "America-NewYork",
    "Japan-Hakodate",
    "Peru-Callao",
]


def build_dirs(cities):
    image_dirs = [os.path.join(root_path, city, "images") for city in cities]
    mask_dirs = [os.path.join(root_path, city, "masks") for city in cities]
    return image_dirs, mask_dirs


train_image_dir, train_mask_dir = build_dirs(train_cities)

val_image_dir, val_mask_dir = build_dirs(val_cities)

test_image_dir, test_mask_dir = build_dirs(test_cities)

