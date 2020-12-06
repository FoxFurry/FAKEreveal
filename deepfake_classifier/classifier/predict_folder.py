import argparse
import os
import re
import time

import torch
import pandas as pd
from deepfake_classifier.classifier.kernel_utils import VideoReader, FaceExtractor, confident_strategy, process_file
from deepfake_classifier.classifier.training.zoo.classifiers import DeepFakeClassifier

torch.cuda.empty_cache()

models = []


def initialize():
    weights_dir = os.path.join(os.getcwd(), "deepfake_classifier")
    weights_dir = os.path.join(weights_dir, "classifier")
    weights_dir = os.path.join(weights_dir, "weights")

    models_name = ["final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
                   "final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19",
                   "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29",
                   "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31",
                   "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37",
                   "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40",
                   "final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23"]

    model_paths = [os.path.join(weights_dir, model) for model in models_name]
    for path in model_paths:
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
        print("loading state dict {}".format(path))
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
        model.eval()
        del checkpoint
        models.append(model.half())

def predict(path):
    current_dir = os.getcwd()

    file_dir = os.path.join(current_dir, path)

    frames_per_video = 32
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn)
    input_size = 380
    strategy = confident_strategy
    stime = time.time()

    predictions = process_file(file = file_dir, face_extractor=face_extractor, input_size=input_size, frames_per_video=frames_per_video, models=models,
                                       strategy=strategy)

    #print("Elapsed:", time.time() - stime)
    return predictions

