import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

def bgr_to_hsv(bgr_2dim):
    normalized_bgr_2dim = np.array(bgr_2dim) / 255.0
    b, g, r = normalized_bgr_2dim[:,0], normalized_bgr_2dim[:,1], normalized_bgr_2dim[:,2]
    
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    diff = max_val - min_val

    # Initialize h, s, and v with zeros
    h = np.zeros_like(r)
    s = np.zeros_like(r)
    v = max_val * 100  # Value

    # Compute Hue
    cond_r = max_val == r
    cond_g = max_val == g
    cond_b = max_val == b
    cond_mx_mn = max_val == min_val

    h = np.where(cond_r, 60 * ((g - b) / diff % 6), h)
    h = np.where(cond_g, 60 * ((b - r) / diff + 2), h)
    h = np.where(cond_b, 60 * ((r - g) / diff + 4), h)
    h = np.where(cond_mx_mn, 0, h)

    # Compute Saturation
    s = np.where(max_val != 0, (diff / max_val) * 100, 0)

    # Stack them into a 2D array
    hsv_2dim = np.stack([h, s, v], axis=1)
    return hsv_2dim

def hsv_to_rgb(hsv_2dim, norm_type="cv2"):
    hsv_2dim = np.array(hsv_2dim).astype(np.float32)

    if norm_type == "cv2":
        scale = np.array([179.0,255.0,255.0])
    else:
        scale = np.array([360.0,100.0,100.0])

    normalized_hsv_2dim = hsv_2dim / scale
    return mcolors.hsv_to_rgb(normalized_hsv_2dim) * 255.0
    
def hsv_to_bgr(hsv_2dim):
    rgb_2dim = hsv_to_rgb(hsv_2dim)
    bgr_2dim = np.array([list(reversed(rgb_1dim)) for rgb_1dim in rgb_2dim])
    return bgr_2dim


def visualize_rgb_colors(rgb_colors):
    rgb_colors = np.array(rgb_colors)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Loop through the list of BGR colors and plot each one
    for i, color in enumerate(rgb_colors):
        # Convert BGR to RGB and normalize to [0, 1]
        rgb_color = [x / 255.0 for x in color]

        # Create a rectangle filled with the normalized RGB color
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=rgb_color)
        
        # Add the rectangle to the plot
        ax.add_patch(rect)

    # Set axis limits and aspect ratio
    ax.set_xlim(0, len(rgb_colors))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
    
def make_pil_rgb_colors(rgb_colors):
    rgb_colors = np.array(rgb_colors)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Loop through the list of BGR colors and plot each one
    for i, color in enumerate(rgb_colors):
        # Convert BGR to RGB and normalize to [0, 1]
        rgb_color = [x / 255.0 for x in color]

        # Create a rectangle filled with the normalized RGB color
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=rgb_color)
        
        # Add the rectangle to the plot
        ax.add_patch(rect)

    # Set axis limits and aspect ratio
    ax.set_xlim(0, len(rgb_colors))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Get the RGB image as a NumPy array
    fig.canvas.draw()
    rgb_image_np = np.array(fig.canvas.renderer.buffer_rgba())

    # Close the plot to free up resources
    plt.close()

    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(rgb_image_np)

    # Return the PIL Image
    return pil_image

def visualize_bgr_colors(bgr_colors):
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Loop through the list of BGR colors and plot each one
    for i, color in enumerate(bgr_colors):
        # Convert BGR to RGB and normalize to [0, 1]
        rgb_color = [x / 255.0 for x in reversed(color)]

        # Create a rectangle filled with the normalized RGB color
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=rgb_color)
        
        # Add the rectangle to the plot
        ax.add_patch(rect)

    # Set axis limits and aspect ratio
    ax.set_xlim(0, len(bgr_colors))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.show()


def visualize_hsv_colors(hsv_colors, norm_type=None):
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    rgb_colors = hsv_to_rgb(hsv_colors, norm_type=norm_type)
    # Loop through the list of HSV colors and plot each one
    for i, rgb_color in enumerate(rgb_colors):
        # Create a rectangle filled with the RGB color
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=rgb_color/255)

        # Add the rectangle to the plot
        ax.add_patch(rect)

    # Set axis limits and aspect ratio
    ax.set_xlim(0, len(hsv_colors))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.show()


from sklearn.decomposition import PCA

def plot_3D_PCA(data, labels=None, label_names=None):
    
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        indices = np.where(labels == label)
        ax.scatter(reduced_data[indices, 0], reduced_data[indices, 1], reduced_data[indices, 2], 
                   label=label_names[label] if label_names else f'Label {label}')

    ax.set_title('3D PCA')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    # Adding the legend
    ax.legend(loc='upper right')

    plt.show()



# clustering kmeans
def img_none_flatten(img, mask):
    if mask is not None:
        img = np.where(mask > 127, img, np.nan)
    return img[~np.isnan(img[:, :, :3]).any(axis=2)]

class Point:
    def __init__(self, data, K=4):
        self.data = data
        self.k = np.random.randint(0, K)

    def __repr__(self):
        return str({"data": self.data, "k": self.k})

def make_k_mapping(points):
    point_dict = defaultdict(list)
    for p in points:
        point_dict[p.k] = point_dict[p.k] + [p.data]
    return point_dict


def calc_k_means(point_dict, K=4):
    means = [np.mean(point_dict[k], axis=0) for k in range(K)]
    return means


def update_k(points, means, K=4):
    for p in points:
        dists = [np.linalg.norm(means[k] - p.data) for k in range(K)]
        p.k = np.argmin(dists)


def find_color_type_fixed(img, mask=None, n_clusters=4, epochs=1):
    img_flatten = img_none_flatten(img.copy(), mask.copy())
    points = [Point(d, K=n_clusters) for d in img_flatten]
    point_dict = make_k_mapping(points)
    colors = calc_k_means(point_dict, K=n_clusters)
    update_k(points, colors, K=n_clusters)
    for e in range(epochs):
        point_dict = make_k_mapping(points)
        colors = calc_k_means(point_dict, K=n_clusters)
        update_k(points, colors, K=n_clusters)

    try:
        colors = [[int(c) for c in color] if isinstance(color, np.ndarray) or isinstance(color, list) else [0, 0, 0] for color in colors]
    except TypeError:
        print(colors)
        raise TypeError

    percentage = [0 for _ in range(n_clusters)]
    for p in points:
        percentage[p.k] += 1
    percentage = [p / len(img_flatten) for p in percentage]

    return colors, percentage


def preprocess_input_image(input_image):
    return input_image

def sort_colors(colors):
    colors.sort(key=lambda c: c[2])
    colors.sort(key=lambda c: c[1])
    colors.sort(key=lambda c: c[0])
    return colors

def sort_colors_hsv(colors):
    if not isinstance(colors, list):
        colors = colors.tolist()
        
    colors.sort(key=lambda c: c[2]**2 + c[1]**2, reverse=True)
    return colors

def sort_color_feature_mean_dist(colors):
    colors = np.array(colors)
    mean_color = np.mean(colors, axis=0)
    dist = np.sum((colors - np.array([mean_color for _ in range(len(colors))])) ** 2, axis=1)/len(colors)
    
    return colors[np.argsort(dist)[::-1]].tolist()


# Kmeans re implementation

class Centroid:
    def __init__(self, color):
        self.data = color
        self.neighbors = np.array([[127,127,127]])

    def update_color(self):
        if len(self.neighbors) == 0:
            self.data = np.array([-255,-255,-255])
        self.data = np.mean(self.neighbors, axis=0)

    def get_dist(self, t_color):
        dist = np.linalg.norm(self.data - t_color)
        return dist
    
    def add_neighbor(self, color):
        self.neighbors = np.append(self.neighbors, color.reshape([1,-1]), axis=0)
    
    def clean_neighbor(self):
        self.neighbors = self.data[None,:]
        
    def get_data(self):
        return self.data.astype(np.int16).tolist(), len(self.neighbors)
    
def random_selected_pixel_with_mask(img, mask=None, select_n=4):
    if mask.all() == None:
        mask = np.ones_like(img[:,:,0])
    if len(mask.shape) == 3:
        mask = mask[:,:,0]
    if len(np.unique(mask)) != 2:
        mask = np.where(mask<127, 0, 1)

    img_flat = img.reshape([-1,3])
    mask_flat = mask.flatten()

    selected_colors = np.array([])

    count = 0
    while(len(np.unique(selected_colors, axis=0)) != select_n):
        selected_pixels = np.random.choice(np.where(mask_flat == 1)[0], select_n, replace=False)    
        selected_colors = img_flat[selected_pixels]
        
        count += 1

        if count >= 30:
            return np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255]])

    return selected_colors

def color_filter_with_mask(img, mask, pixel_skip):
    if mask.all() == None:
        mask = np.ones_like(img[:,:,0])
    if len(mask.shape) == 3:
        mask = mask[:,:,0]
    if len(np.unique(mask)) != 2:
        mask = np.where(mask<127, 0, 1)

    img_flat = img.reshape([-1,3])
    mask_flat = mask.flatten()
    img_flat = img_flat[np.where(mask_flat == 1)[0]]
    random_row = np.random.choice(len(img_flat), size=len(img_flat)//pixel_skip, replace=True)

    return img_flat[random_row,:]


def color_extraction(img, mask=None, n_cluster=4, epochs = 3, pixel_skip=10, per_round=None):
    if mask is None:
        mask = np.ones_like(img) * 255

    img = color_filter_with_mask(img, mask, pixel_skip)

    # selected_colors = random_selected_pixel_with_mask(img, mask, n_cluster)
    if len(img) == 0:
        print(np.unique(mask))
    selected_colors = img[np.random.choice(range(len(img)), n_cluster)]
    k_map = {idx:Centroid(color) for idx, color in enumerate(selected_colors)}

    result = []
    for epoch in range(epochs):
        for color in img:
            closest_k = np.argmin([k_map[k].get_dist(color) for k in range(n_cluster)])
            k_map[closest_k].add_neighbor(color)

        for k in range(n_cluster):
            k_map[k].update_color()

            if epoch == epochs-1:
                color, num_pix = k_map[k].get_data()
                result.append((color, num_pix/len(img)))
            else:
                k_map[k].clean_neighbor()

    result = sorted(result, key=lambda x:x[1], reverse=True)

    color_result = []
    percentage = []
    for cur in result:
        color_result.append(cur[0])
        if per_round is not None:
            per = round(cur[1], per_round)
        else:
            per = cur[1]
        percentage.append(per)

    return [color_result, percentage]



def color_normalization(color_arr, scaling = True, type="rgb", only_scale=False):

    color_arr = np.array(color_arr).astype(np.float32)

    if type == "rgb":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        scale = np.array([255.0, 255.0, 255.0])
    elif type == "hsv":
        mean = np.array([0.314 , 0.3064, 0.553])
        std = np.array([0.2173, 0.2056, 0.2211])
        scale = np.array([179.0, 255.0, 255.0])
        
    arr_shape_length = len(color_arr.shape)    
    new_shape = [1]*arr_shape_length
    new_shape[arr_shape_length-1] = -1

    mean = mean.reshape(new_shape)
    std = std.reshape(new_shape)
    scale = scale.reshape(new_shape)

    if scaling == True:
        color_arr /= scale

    if only_scale == True:
        return color_arr
    
    return (color_arr - mean)/std


def color_normalization_restore(color_arr, scaling = True, type="rgb"):
    color_arr = np.array(color_arr).astype(np.float32)

    if type == "rgb":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        scale = np.array([255.0, 255.0, 255.0])
    elif type == "hsv":
        mean = np.array([0.314 , 0.3064, 0.553])
        std = np.array([0.2173, 0.2056, 0.2211])
        scale = np.array([179.0, 255.0, 255.0])
        
    arr_shape_length = len(color_arr.shape)
    new_shape = [1]*arr_shape_length
    new_shape[arr_shape_length-1] = -1
 
    mean = mean.reshape(new_shape)
    std = std.reshape(new_shape)
    scale = scale.reshape(new_shape)

    restored_arr = color_arr * std + mean

    if scaling == True:
        restored_arr *= scale

    return restored_arr

import cv2
def rgb2hsv_cv2(rgb_colors):
    rgb_colors = np.array(rgb_colors, dtype=np.uint8)

    input_shape_len = len(rgb_colors.shape)
    if input_shape_len < 3:
        rgb_colors = rgb_colors.reshape(1,-1,3)
    
    hsv_colors = cv2.cvtColor(rgb_colors, cv2.COLOR_RGB2HSV).astype(np.uint8)

    return hsv_colors.squeeze()

def hsv2rgb_cv2(hsv_colors):
    hsv_colors = np.array(hsv_colors, dtype=np.uint8)

    input_shape_len = len(hsv_colors.shape)
    if input_shape_len < 3:
        hsv_colors = hsv_colors.reshape(1,-1,3)
    
    rgb_colors = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2RGB).astype(np.uint8)

    return rgb_colors.squeeze()


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

def colors_to_hex(colors):
    colors = list(colors)
    return [rgb_to_hex(color) for color in colors]


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / max(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

from PIL import Image
if __name__ == "__main__":

    bgr_2dim = np.array([[28,42,42], [0,0,255], [10,20,200]])
    visualize_rgb_colors(bgr_2dim)
    
    hsv_color = rgb2hsv_cv2(bgr_2dim)
    print(hsv_color)
    norm_color = color_normalization_restore(color_normalization(hsv_color, type="hsv"), type="hsv")
    print(norm_color.astype(np.uint8))

    visualize_hsv_colors(hsv_color, norm_type="cv2")

    # hsv_2dim = bgr_to_hsv(bgr_2dim)
    # rgb_2dim = hsv_to_rgb(hsv_2dim)
    # bgr_2dim_2 = hsv_to_bgr(hsv_2dim)

    # print("BGR Colors")
    # print(bgr_2dim)
    # visualize_bgr_colors(bgr_2dim)

    # print("HSV Colors")
    # print(hsv_2dim)
    # visualize_hsv_colors(hsv_2dim)

    # print("RGB Colors")
    # print(rgb_2dim)
    # visualize_rgb_colors(rgb_2dim)

    # print("BGR Colors2")
    # print(bgr_2dim_2)
    # visualize_bgr_colors(bgr_2dim)