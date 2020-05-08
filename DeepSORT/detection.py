#!/usr/bin/env python3

import numpy as np

class Detection():

    """
    This class represents a bounding box detection in a single image.

    Parameters:
        bbox:           Bounding box in (top left x, top left y, w, h) format
        confidence:     Detector confidence score
        feature:        Feature vector that describes the object contained in the image
    """

    def __init__(self, bbox, confidence, feature):
        self.bbox_ = np.asarray(bbox, dtype=np.float)
        self.confidence_ = confidence
        self.feature_ = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """ Convert bounding box to (top left, bottom right) format
        """

        tlbr = self.bbox_.copy()
        tlbr[2:] += tlbr[:2]

        return tlbr

    def to_xyah(self):
        """ Convert bounding box to (center_x, center_y, aspect_ratio, height)
            format, where aspect_ratio = width/height
        """

        xyah = self.bbox_.copy()
        xyah[:2] += xyah[2:]/2
        xyah[2] /= xyah[3]

        return xyah
