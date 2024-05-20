"""
This file contains all configuration parameters
later used throghout the project's code
"""

import math

class Composition(object):
    """
    Parameters composition
    """

    VERBOSE = True
    NETWORK = 'vgg'

    # Data augmentation parameters
    HORIZ_FLIPS = False
    VERT_FLIPS = False
    ROTATE_90 = False

    # Anchor box scales
    ANCHOR_BOX_SCALES = [16, 32, 64]
    ANCHOR_BOX_RATIOS = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)],
                         [2./math.sqrt(2), 1./math.sqrt(2)]]

    # Resize input image to the following size
    IMG_RESIZE_TO = 300

    # image channel-wise mean to subtract
    IMG_CHANNEL_MEAN = [103.939, 116.779, 123.68]
    IMG_SCALING_FACTOR = 1.0

    # Set the number of ROIS
    NUM_ROIS = 4

    # Stride at the RPN (this depends on the network architecture, i.e., 16)
    NUM_STRIDE = 16

    # Determine wether we have balaced or unbalanced classes
    BALANCED_CLASSES = False

    # Scaling the stdev
    STD_SCALING = 4.0
    CLASSIFIER_REGR_STD = [8.0, 8.0, 4.0, 4.0]

    # Overlaps for RPN
    RPN_MIN_OVERLAP = 0.3
    RPN_MAX_OVERLAP = 0.7

    # Overlaps for classifier ROIs
    CLASSIFIER_MIN_OVERLAP = 0.1
    CLASSIFIER_MAX_OVERLAP = 0.5

    # Placeholder for the class mapping, automatically generated by the parser
    CLASS_MAPPING = None

    MODEL_PATH = None

    def show_composition(self):
        """
        Show system configuration
        """
        
