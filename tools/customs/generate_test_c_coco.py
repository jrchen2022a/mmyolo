from os.path import join, exists
from os import mkdir
from json import load as json_load
from shutil import copyfile


def extract_imgs(src_path, dst_path):
    anno_path = join(src_path,'annotations/test.json')
    src_images_path = join(src_path,'images')
    src_labels_path = join(src_path,'labels')
    if not exists(anno_path) or not exists(src_images_path) or not exists(src_labels_path):
        return
    if not exists(dst_path):
        mkdir(dst_path)
    dst_images_path = join(dst_path, 'originals')
    dst_labels_path = join(dst_path, 'labels')
    if not exists(dst_images_path):
        mkdir(dst_images_path)
    if not exists(dst_labels_path):
        mkdir(dst_labels_path)

    with open(anno_path, 'r') as f:
        coco_data = json_load(f)

    images = coco_data['images']
    for item in images:
        img = item['file_name']
        src_img_path = join(src_images_path, img)
        if not exists(src_img_path):
            print("---- {} not exist ---".join(src_img_path))
            continue
        label = img.split('.')[0]+'.txt'
        copyfile(src_img_path, join(dst_images_path, img))
        src_label_path = join(src_labels_path, label)
        if exists(src_label_path):
            copyfile(src_label_path, join(dst_labels_path, label))


if __name__ == '__main__':
    src_path = r'/home/jrchen/datasets/researches'
    dst_path = join(src_path, 'corruptions')
    # extract_imgs(src_path, dst_path)
    from generate_test_c import generate_c, generate_c_standard
    # generate_c(join(dst_path, 'originals'), dst_path, corruptions=['elastic','contrast'])
    generate_c_standard(join(dst_path, 'originals'), join(src_path, 'corruptions1'))

