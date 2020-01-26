from imgaug.augmenters import Resize


class ResizeToShape(Resize):
    """
    Wrapper augmenter which ensures that resizing always chooses the smallest axis for keeping the aspect ratio.
    In other words, the bigger axis is rezized to the specified target dimension while maintaining the aspect ratio
    for the smaller axis.

    This ensures that subsequent padding is enough for getting to the target resolution without the need to crop.
    """
    def _compute_height_width(self, image_shape, height, width, dim_order='SL'):
        ratio = image_shape[0]/image_shape[1]
        if (height / width) / ratio < 1:
            h = height
            w = "keep-aspect-ratio"
        else:
            h = "keep-aspect-ratio"
            w = width

        return super(ResizeToShape, self)._compute_height_width(image_shape, h, w, dim_order)

