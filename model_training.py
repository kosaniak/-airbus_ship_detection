import numpy as np
import pandas as pd
import imageio.v2 as imageio
from imageio import imread
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

df = pd.read_csv('train_ship_segmentations_v2.csv')
print(df.shape)
df.head()

# Data balance checking
ships = df[~df.EncodedPixels.isna()].ImageId.unique()
noships = df[df.EncodedPixels.isna()].ImageId.unique()

plt.bar(['Ships', 'No Ships'], [len(ships), len(noships)], color=['green', 'blue']);
plt.ylabel('Number of Images');

# Delete missing values
masks = df.drop(df[df['EncodedPixels'].isnull()].sample(70000,random_state=42).index)

masks.shape

# Grouping by the number of ships that belong to one picture
unique_img_ids = masks.groupby('ImageId').size().reset_index(name='Counts')

# Split into training and validation groups
train_ids, valid_ids = train_test_split(unique_img_ids, 
                 test_size = 0.2, 
                 stratify = unique_img_ids['Counts'],
                 random_state=42)


# Merge of samples, one image can correspond to several lines
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0],  'training masks')
print(valid_df.shape[0],  'validation masks')

train_df.head()
valid_df.head()

# Replace NaN with zero
train_df['Counts'] = train_df.apply(lambda c_row: c_row['Counts'] if 
                                    isinstance(c_row['EncodedPixels'], str) else
                                    0, 1)
valid_df['Counts'] = valid_df.apply(lambda c_row: c_row['Counts'] if 
                                    isinstance(c_row['EncodedPixels'], str) else
                                    0, 1)

# The ratio of the num of ships in sample
train_df['Counts'].hist(bins=15);
valid_df['Counts'].hist(bins=15);

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

# Example
ImageId = '194a2b8c9.jpg'

img = imread('../new/train_v2/' + ImageId)
img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

all_masks = np.zeros((768, 768))
for mask in img_masks:
    all_masks += rle_decode(mask)

fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')
axarr[0].imshow(img[...,[2,1,0]]) #rgb
axarr[1].imshow(all_masks, cmap='viridis')
axarr[2].imshow(img, cmap='viridis')
axarr[2].imshow(all_masks, alpha=0.4, cmap='viridis')
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()

# Make a generator to produce batches of images
IMG_SCALING = (1,1)

def keras_generator(gen_df, batch_size=4):
    all_batches = list(gen_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join('../new/train_v2', c_img_id)
            c_img = cv2.imread(rgb_path)
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []

# Checking the dimension
train_gen = keras_generator(train_df,5)
train_x, train_y = next(train_gen)
print('x', train_x.shape, train_x.dtype, train_x.min(), train_x.max())
print('y', train_y.shape, train_y.dtype, train_y.min(), train_y.max())
