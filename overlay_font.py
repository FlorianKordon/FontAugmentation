# std
import glob
import string
from os.path import join

# image handling
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import img_as_float, img_as_ubyte

# imgaug
import imgaug.augmenters as iaa
from imgaug import parameters as iap

# typing
from typing import Tuple, Union
try:
    from typing import TypedDict
except ImportError:
    from mypy_extensions import TypedDict

EPS = 1e-17


def minmax_normalize_images(img):
    return (img - img.min() + EPS) / (img.max() - img.min() + EPS)


class VectorParameter(TypedDict):
    x: Tuple[Union[int, float], Union[int, float]]
    y: Tuple[Union[int, float], Union[int, float]]


class OverlayFont(iaa.Augmenter):
    def __init__(self,
                 dir_fonts: str,
                 ext_fonts: list = ['ttf', 'otf'],
                 target_height: int = 256,
                 target_width: int = 256,
                 num_samples: Tuple[int, int] = (1, 2),
                 length_samples: Tuple[int, int] = (1, 2),
                 overlay_scale: VectorParameter = {"x":(0.05, 2), "y":(0.05, 2)},
                 overlay_shift: VectorParameter = {"x": (-50, 50), "y": (-50, 50)},
                 overlay_rotation: Tuple[int, int] = (0, 360),
                 overlay_intensity: Tuple[float, float] = (0.5, 0.9),
                 name: str = None,
                 deterministic: bool = False,
                 random_state: np.random.RandomState = None
                 ) -> None:
        super(OverlayFont, self).__init__(name=name,
                                          deterministic=deterministic,
                                          random_state=random_state)
        self.fonts_fns = []
        for ext in ext_fonts:
            self.fonts_fns += glob.glob(join(dir_fonts, '*') + '.' + ext)
        assert len(self.fonts_fns) > 0

        self.target_height = target_height
        self.target_width = target_width

        # Setup font and symbol sampling parameter
        self.font_id = iap.handle_discrete_param(param=(0, len(self.fonts_fns) - 1),
                                                 name="font_id",
                                                 value_range=None,
                                                 tuple_to_uniform=True,
                                                 list_to_choice=True,
                                                 allow_floats=False)

        self.symbols = list(string.ascii_letters) + list(string.digits)
        self.symbol_id = iap.handle_discrete_param(param=(0, len(self.symbols) - 1),
                                                   name="character_id",
                                                   value_range=None,
                                                   tuple_to_uniform=True,
                                                   list_to_choice=True,
                                                   allow_floats=False)

        self.num_samples = (
            iap.handle_discrete_param(param=num_samples,
                                      name="num_samples",
                                      value_range=None,
                                      tuple_to_uniform=True,
                                      list_to_choice=True,
                                      allow_floats=False)
        )
        self.max_num_samples = max(num_samples)
        self.length_samples = length_samples
        self.max_length_samples = max(self.length_samples)
        if self.length_samples is not None:
            self.length_samples = (
                iap.handle_discrete_param(param=length_samples,
                                          name="length_samples",
                                          value_range=None,
                                          tuple_to_uniform=True,
                                          list_to_choice=True,
                                          allow_floats=False)
            )

        # Setup imgaug parameter for symbol intensity
        self.overlay_intensity = (
            iap.handle_continuous_param(param=overlay_intensity,
                                        name="implant_intensity",
                                        value_range=None,
                                        tuple_to_uniform=True,
                                        list_to_choice=True)
        )
        self.overlay_augmenter = self.get_overlay_augmenter(random_state=random_state,
                                                            scale=overlay_scale,
                                                            shift=overlay_shift,
                                                            rotation=overlay_rotation)

    def get_overlay_augmenter(self,
                              random_state: np.random.RandomState,
                              scale: VectorParameter,
                              shift: VectorParameter,
                              rotation: Tuple[int, int]
                              ) -> iaa.Augmenter:

        def _img_func_mask(images, random_state, parents, hooks):
            for img in images:
                img[::4] = 0
            return images

        def _hm_func_mask(hms, random_state, parents, hooks):
            return hms

        def _keypoint_func_mask(keypoints_on_images, random_state, parents, hooks):
            return keypoints_on_images

        overlay_augmenter = iaa.Sequential([
            # small local distortions
            iaa.Sometimes(p=0.5, then_list=[iaa.PiecewiseAffine(scale=(0.01, 0.05))]),
            # structured occlusion to emulate screw patterns
            iaa.Sometimes(p=0.1, then_list=[iaa.Lambda(_img_func_mask, _hm_func_mask, _keypoint_func_mask)]),
            # affine transformations
            iaa.Affine(
                scale=scale,
                translate_px=shift,
                rotate=rotation,
                order=0                 # use nearest neighbour
            ),
            iaa.GaussianBlur(sigma=1),
        ], random_order=False, random_state=random_state)
        return overlay_augmenter

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        result = []

        # sample imgaug parameters with given random state
        num_seq_samples = self.num_samples.draw_samples(size=(nb_images,),
                                                        random_state=random_state)
        len_seq_samples = self.length_samples.draw_samples(size=(nb_images, self.max_num_samples),
                                                           random_state=random_state)
        font_id_samples = self.font_id.draw_samples(size=(nb_images, self.max_num_samples),
                                                    random_state=random_state)
        intensity_samples = self.overlay_intensity.draw_samples(size=(nb_images, self.max_num_samples),
                                                                random_state=random_state)
        character_id_samples = self.symbol_id.draw_samples(size=(nb_images, self.max_num_samples, self.max_length_samples),
                                                           random_state=random_state)

        for n in range(nb_images):
            img = images[n]

            # process and overlay samples
            overlays = []
            for j in range(num_seq_samples[n]):
                font = ImageFont.truetype(font=self.fonts_fns[font_id_samples[n, j]], size=2048)
                text = ""
                for s in range(len_seq_samples[n, j]):
                    text += self.symbols[character_id_samples[n, j, s]]
                font_w, font_h = font.getsize(text)
                font_h += int(font_h * 0.21)

                # draw characters
                canvas = Image.new("L", (font_w * 2, font_h * 2), 0)
                drawer = ImageDraw.Draw(canvas)
                drawer.text(xy=((canvas.size[0] - font_w) / 2, (canvas.size[1] - font_h) / 2),
                            text=text,
                            fill=256,
                            font=font)

                # pad
                max_dim = max(canvas.size)
                padded = Image.new("L", (max_dim, max_dim))
                padded.paste(canvas, ((max_dim - canvas.size[0]) // 2, (max_dim - canvas.size[1]) // 2))

                img_overlay = np.array(padded.resize((self.target_width, self.target_height), Image.BILINEAR))
                img_overlay = self.overlay_augmenter.augment_image(img_overlay)

                # truncate the overlay values to 1 when target intensity is 1
                target_intensity = intensity_samples[n, j]
                if target_intensity == 1:
                    img_overlay = img_as_float(img_overlay)
                    img_overlay[img_overlay > EPS] = 1.
                else:
                    img_overlay = img_as_float(img_overlay) * target_intensity

                # overlay image
                img = img.squeeze() * (1 - img_overlay)

                overlays.append(img_overlay)

            # normalize
            img_overlayed = minmax_normalize_images(img)

            # add channel axis to fit imgaug processing format
            img_overlayed = np.expand_dims(img_overlayed, axis=-1)

            # convert back to ubyte format
            img_overlayed = img_as_ubyte(img_overlayed)

            result.append(img_overlayed)
        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return self.heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.font_id, self.symbol_id, self.implant_shift, self.implant_rotation, self.overlay_intensity]
