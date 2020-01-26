if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import imgaug.augmenters as iaa
    import glob

    from skimage import io, img_as_ubyte
    from overlay_font import OverlayFont, minmax_normalize_images
    from resize_to_shape import ResizeToShape

    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=bool, help='Run augmentation on empty background',
                        default=1)
    parser.add_argument("--img_dir", type=str, help='Set source image directory')
    parser.add_argument("--img_ext", nargs='+', help='Set image file extenion(s)',
                        default=['jpg', 'png'])
    parser.add_argument("--font_dir", type=str, help='Set source font directory', required=False)
    parser.add_argument("--font_ext", nargs='+', help='Set font file extenion(s)',
                        default=['ttf', 'otf'])
    parser.add_argument("--target_height", type=int, help='Set target image height',
                        default=1024)
    parser.add_argument("--target_width", type=int, help='Set target image width',
                        default=1024)

    args, unknown_args = parser.parse_known_args()
    if not args.example and (args.img_dir is None):
        parser.error("--img_dir is required if example mode is set to false.")

    example = args.example
    img_dir = args.img_dir
    img_ext = args.img_ext
    font_dir = args.font_dir
    font_ext = args.font_ext
    target_height = args.target_height
    target_width = args.target_width

    augmenter = iaa.Sequential([
        ResizeToShape({"height": target_height, "width": target_width}),
        iaa.PadToFixedSize(height=target_height, width=target_width, position='center'),
        OverlayFont(dir_fonts=font_dir,
                    ext_fonts=font_ext,
                    target_height=target_height,
                    target_width=target_width),
        ], random_order=False)

    if example:
        print("Running example with augmentation on sample image background.")
        fns_img = ['./images/example_img.jpg' for _ in range(10)]
    else:
        fns_img = []
        for ext in img_ext:
            fns_img += glob.glob(f"{img_dir}/*.{ext}")
        if len(fns_img) == 0:
            print("No images found in the provided directory.")

    for i, fn in enumerate(fns_img):
        # convert image to uint8
        img = img_as_ubyte(io.imread(fn, as_gray=True).squeeze())
        # normalize the image (min-max normalization)
        img = minmax_normalize_images(img)
        # augment the image with the specified augmentation pipeline
        img_aug = augmenter.augment_image(img)
        # second image normalization
        img_aug = minmax_normalize_images(img_aug)
        # optional: convert back to ubyte
        img_aug = img_as_ubyte(img_aug)

        # visualization
        fig = plt.figure()

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title('Original image')
        plt.axis("off")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(img_aug, cmap=plt.cm.gray)
        ax2.set_title('Augmented image')
        plt.axis("off")
        plt.show()
