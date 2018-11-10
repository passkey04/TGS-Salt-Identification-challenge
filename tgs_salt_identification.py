import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils import data
%matplotlib inline
from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

class TGS(data.Dataset):
  def __init__(self,root_path,file_list):
    self.root_path=root_path
    self.file_list=file_list
  def __len__(self):
      return len(self.file_list)
  def __getitem__(self,index):
    if index not in range(0,len(self.file_list)):
      return self.__getitem__(np.random.randint(0,self.__len__())) #if the index is out of range, retirn a rand image.
    file_id = self.file_list[index]
    image_folder=os.path.join(self.root_path,"images")
    image_path = os.path.join(image_folder, file_id + ".png")
    mask_folder=os.path.join(self.root_path,"masks")
    mask_path=os.path.join(mask_folder,file_id+".png")
    image=np.array(imageio.imread(image_path),dtype=np.uint8)#store the images in an array
    mask=np.array(imageio.imread(mask_path),dtype=np.uint8)
    return image,mask

"""Training image and mask data read"""

train_mask=pd.read_csv('train.csv')
depth=pd.read_csv('depths.csv')
train_path="./"
file_list=list(train_mask['id'].values)
dataset=TGS(train_path,file_list)

"""Visualization of images"""

def vizarr(image, mask):
    x, arr = plt.subplots(1,2)
    arr[0].imshow(image)
    arr[1].imshow(mask)
    arr[0].grid()
    arr[1].grid()
    arr[0].set_title('Image')
    arr[1].set_title('Mask')

for i in range(0,5):
    image, mask = dataset[np.random.randint(0, len(dataset))]
    vizarr(image, mask)

plt.figure(figsize = (10, 10))
plt.hist(depth['z'], bins = 50)
plt.title('Distribution of depths')

"""Now for data compression- Run length encoding is used.
This function will convert to image
"""

def rletomask(rlestring,height,width):
  rows,cols=height,width
  try:
    rleNumbers=[int(numstring) for numstring in rlestring.split(' ')]
    rlePairs=np.array(rle.Numbers).reshape(1,-2)
    img=np.zeros(rows*cols,dtype=unit8)
    for index,length in rlePairs:#get pixel value for each rlepairs
      index-=-1
      img[index:index+length]=255
    img=img.reshape(cols,rows)
    img=img.T
  #exception, returns an empty image
  except:
    img=np.zeros((cols,rows))
  return img

"""Measuring salt quantity in the image"""

def salt_proportion(imgArray):
    try: 
        unique, counts = np.unique(imgArray, return_counts=True)
        return counts[1]/10201.
    
    except: 
        return 0.0

"""Merging the depths with other data frame objects"""

train_mask['mask'] = train_mask['rle_mask'].apply(lambda x: rletomask(x, 101,101))
train_mask['salt_proportion'] = train_mask['mask'].apply(lambda x: salt_proportion(x))

merged = train_mask.merge(depth, how = 'left')
merged.head()

plt.figure(figsize=(10,10))
plt.scatter(merged['salt_proportion'], merged['z'])
plt.title('Proportion of salt vs. depth')

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import sys
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize

im_width = 128
im_height = 128
border = 5
im_chan = 2
n_features = 1

"""U-net model"""

input_img = Input((im_height, im_width, im_chan), name='img')
input_features = Input((n_features, ), name='feat')
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (input_img)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
f_repeat = RepeatVector(8*8)(input_features)
f_conv = Reshape((8, 8, n_features))(f_repeat)
p4_feat = concatenate([p4, f_conv], -1)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4_feat)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)
u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)
u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)
u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)
u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)
outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
model = Model(inputs=[input_img, input_features], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy') #, metrics=[mean_iou]) # The mean_iou metrics seens to leak train and test values...
model.summary()

train_ids = next(os.walk(train_path+"images"))[2]
X = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.float32)
y = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.float32)
X_feat = np.zeros((len(train_ids), n_features), dtype=np.float32)
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = train_path
    img = load_img(path + '/images/' + id_, grayscale=True)
    x_img = img_to_array(img)
    x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)
    x_center_mean = x_img[border:-border, border:-border].mean()
    x_csum = (np.float32(x_img)-x_center_mean).cumsum(axis=0)
    x_csum -= x_csum[border:-border, border:-border].mean()
    x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())
    mask = img_to_array(load_img(path + '/masks/' + id_, grayscale=True))
    mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)
    X[n, ..., 0] = x_img.squeeze() / 255
    X[n, ..., 1] = x_csum.squeeze()
    y[n] = mask / 255
print('Completed')

from sklearn.model_selection import train_test_split
X_train, X_valid, X_feat_train, X_feat_valid, y_train, y_valid = train_test_split(X, X_feat, y, test_size=0.15, random_state=42)

callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

results = model.fit({'img': X_train, 'feat': X_feat_train}, y_train, batch_size=16, epochs=50, callbacks=callbacks,
                    validation_data=({'img': X_valid, 'feat': X_feat_valid}, y_valid))

