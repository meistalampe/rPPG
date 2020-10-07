from hdf5_helper_functions import images_to_video


def main():
    # Normalized Thermal Images
    img_folder = 'E:\\GitHub\\CovPySourceFile\\Normalized\\'
    name_tag = 'NF'
    # Multi-Level Otsu Masks
    # img_folder = 'E:\\GitHub\\CovPySourceFile\\MultiLevelOtsu\\'
    # name_tag = 'MLO'
    # Unsharpened Thermal Images
    # img_folder = 'E:\\GitHub\\CovPySourceFile\\UnsharpenedMask\\'
    # name_tag = 'UM'
    # Demo with ROI
    img_folder = 'E:\\GitHub\\CovPySourceFile\\DemoVideo\\'
    name_tag = 'DF'

    destination_dir = 'E:\\GitHub\\CovPySourceFile\\Video\\'

    images_to_video(
        image_dir=img_folder,
        target_dir=destination_dir,
        image_name_tag=name_tag,
        file_type='.png',
        video_name=name_tag + '_video'
    )


if __name__ == '__main__':
    main()
