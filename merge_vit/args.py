import os
import argparse

import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ViT-B-32",
        help="The type of model (e.g. ViT-B-32, ViT-L-14).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--drop-ratio",
        type=float,
        default=0.0,
        help="test alpha."
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=64,
        help="ratio of scale param."
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=10,
        help='the number of knn neighbor'
    )
    parser.add_argument(
        "--knn-pre-sample-num",
        type=int,
        default=100,
        help='knn sample num'
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"    
    return parsed_args, parser
