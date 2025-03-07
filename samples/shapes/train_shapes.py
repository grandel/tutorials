# !/usr/bin/env python
# coding: utf-8

import argparse
import os
import random
import sys

import numpy as np
from matplotlib import pyplot as plt

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from mrcnn.datasets import CocoLikeDataset
from mrcnn.configs import CocoLikeInferenceConfig, CocoLikeConfig
import imgaug

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def display_random_images(dataset_train):
    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


def create_model(model_dir, config, coco_model_path, mode="training"):
    # Create model in training mode
    model = modellib.MaskRCNN(mode=mode, config=config,
                              model_dir=model_dir)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(coco_model_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
    return model


def train_heads(model, dataset_train, dataset_val, config):
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    augmentation = imgaug.augmenters.Sometimes(0.5, [
        imgaug.augmenters.Fliplr(0.5),
        imgaug.augmenters.PerspectiveTransform(),
        imgaug.augmenters.ElasticTransformation(),
        imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
    ])
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=80, layers='heads',
                augmentation=augmentation)
    return model


def train_all(model, dataset_train, dataset_val, config):
    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=20, layers="3+")
    return model


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually


def save_model(model, model_dir, model_name="mask_rcnn_mga.h5"):
    model_path = os.path.join(model_dir, model_name)
    model.keras_model.save_weights(model_path)


def infere(config, model_dir, dataset_train, dataset_val):
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=model_dir)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    model_path = os.path.join("./../../logs/mga-logo20210529T0117/", "mask_rcnn_mga-logo_0080.h5")
    # model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Test on a random image
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, config,
                                                                                       image_id)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names,
                                figsize=(8, 8))

    results = model.detect([original_image], verbose=1)

    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names,
                                r['scores'], ax=get_ax())


def infere_single_image(config, model_dir, dataset_train, input_dir):
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=model_dir)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    model_path = os.path.join("./logs/mga-logo20210529T0117/", "mask_rcnn_mga-logo_0080.h5")
    # model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    for file in os.listdir(input_dir):
        file_name = os.path.join(input_dir, file)
        original_image = load_img(file_name)
        original_image = img_to_array(original_image)
        results = model.detect([original_image], verbose=1)
        r = results[0]
        visualize.save_instances(file_name, original_image, r['rois'], r['masks'], r['class_ids'],
                                 dataset_train.class_names,
                                 r['scores'], ax=get_ax())


# ## Evaluation

def evalute(model, config, dataset_val):
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = np.random.choice(dataset_val.image_ids, 10)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, config,
                                                                                  image_id)
        molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

    print("mAP: ", np.mean(APs))


def main():
    # Directory to save logs and trained model
    model_dir = os.path.join(ROOT_DIR, "logs")
    #
    # # Local path to trained weights file
    coco_model_path = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # # Download COCO trained weights from Releases if needed
    if not os.path.exists(coco_model_path):
        utils.download_trained_weights(coco_model_path)

    # config = CocoLikeConfig()
    # config.display()
    #
    #
    logo_dataset_dir = "./../../datasets/mga"
    logo_dataset_train_json = os.path.join(logo_dataset_dir, "train", "logo_train_coco.json")
    logo_dataset_train_images = os.path.join(logo_dataset_dir, "train", "images")
    dataset_train = CocoLikeDataset()
    dataset_train.load_data(logo_dataset_train_json, logo_dataset_train_images)
    dataset_train.prepare()

    # Validation dataset
    logo_dataset_val_json = os.path.join(logo_dataset_dir, "val", "logo_val_coco.json")
    logo_dataset_val_images = os.path.join(logo_dataset_dir, "val", "images")
    dataset_val = CocoLikeDataset()
    dataset_val.load_data(logo_dataset_val_json, logo_dataset_val_images)
    dataset_val.prepare()

    # display_random_images(dataset_train)

    # last_coco_model_path = "./../../logs/mga-logo20210503T2026"
    # model = create_model(model_dir, config, last_coco_model_path)
    # model = create_model(model_dir, config, model_dir)
    # model = train_heads(model, dataset_train, dataset_val, config)
    # model = train_all(model, dataset_train, dataset_val, config)
    # save_model(model, model_dir)
    inference_config = CocoLikeInferenceConfig()
    inference_config.display()
    # infere_model_dir = model_dir
    # infere(inference_config, infere_model_dir, dataset_train, dataset_val)

    # last_coco_model_path = "./../../logs/mga-logo20210503T2026"
    last_coco_model_path = model_dir
    model = create_model(model_dir, inference_config, last_coco_model_path, "inference")
    evalute(model, inference_config, dataset_val)


def inference():
    logo_dataset_dir = "./datasets/mga3"
    logo_dataset_train_json = os.path.join(logo_dataset_dir, "train", "logo_train_coco.json")
    logo_dataset_train_images = os.path.join(logo_dataset_dir, "train", "images")
    dataset_train = CocoLikeDataset()
    dataset_train.load_data(logo_dataset_train_json, logo_dataset_train_images)
    dataset_train.prepare()

    # Validation dataset
    logo_dataset_val_json = os.path.join(logo_dataset_dir, "val", "logo_val_coco.json")
    logo_dataset_val_images = os.path.join(logo_dataset_dir, "val", "images")
    dataset_val = CocoLikeDataset()
    dataset_val.load_data(logo_dataset_val_json, logo_dataset_val_images)
    dataset_val.prepare()

    model_dir = os.path.join(ROOT_DIR, "logs")
    inference_config = CocoLikeInferenceConfig()
    inference_config.display()
    # infere_model_dir = model_dir
    infere_model_dir = "./../../logs"
    infere_single_image(inference_config, infere_model_dir, dataset_train, "./inference/input")


def train():
    model_dir = os.path.join(ROOT_DIR, "logs")
    coco_model_path = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # # Download COCO trained weights from Releases if needed
    # if not os.path.exists(coco_model_path):
    #     utils.download_trained_weights(coco_model_path)
    config = CocoLikeConfig()
    config.display()
    logo_dataset_dir = "./datasets/mga3"
    logo_dataset_train_json = os.path.join(logo_dataset_dir, "train", "logo_train_coco.json")
    logo_dataset_train_images = os.path.join(logo_dataset_dir, "train", "images")
    dataset_train = CocoLikeDataset()
    dataset_train.load_data(logo_dataset_train_json, logo_dataset_train_images)
    dataset_train.prepare()

    # Validation dataset
    logo_dataset_val_json = os.path.join(logo_dataset_dir, "val", "logo_val_coco.json")
    logo_dataset_val_images = os.path.join(logo_dataset_dir, "val", "images")
    dataset_val = CocoLikeDataset()
    dataset_val.load_data(logo_dataset_val_json, logo_dataset_val_images)
    dataset_val.prepare()

    model = create_model(model_dir, config, coco_model_path)
    model = train_heads(model, dataset_train, dataset_val, config)
    # model = train_all(model, dataset_train, dataset_val, config)


def evaluate():
    model_dir = os.path.join(ROOT_DIR, "logs")
    inference_config = CocoLikeInferenceConfig()
    inference_config.display()

    logo_dataset_dir = "./datasets/mga"
    logo_dataset_val_json = os.path.join(logo_dataset_dir, "val", "logo_val_coco.json")
    logo_dataset_val_images = os.path.join(logo_dataset_dir, "val", "images")
    dataset_val = CocoLikeDataset()
    dataset_val.load_data(logo_dataset_val_json, logo_dataset_val_images)
    dataset_val.prepare()

    last_coco_model_path = "./logs/mga-logo20210505T1214"
    model = create_model(model_dir, inference_config, last_coco_model_path, "inference")
    evalute(model, inference_config, dataset_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action="store_true")
    group.add_argument('--infere', action="store_true")
    args = parser.parse_args()

    # main()
    if args.train:
        train()
    if args.infere:
        inference()

    # evaluate()
    pass
