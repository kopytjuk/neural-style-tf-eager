import imageio
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

fonts_dir = os.path.join(os.environ['WINDIR'], 'Fonts')
font_name = 'consolab.ttf'
font = ImageFont.truetype(os.path.join(fonts_dir, font_name), size=25)


def create_training_gif(outpath, content_path, style_path, iter_path, size=512):
    
    image_size = (int(size*1.5), int(size*3 + 4*size/4), 3)
    image_size = np.asarray(image_size, dtype=int)

    content_bbox = (size/4, size/4, size/4+size, size/4+size)
    style_bbox = (image_size[1] - size/4 - size, size/4, image_size[1] - size/4, size/4+size)
    iter_bbox = (size/4 + size + size/4 , size/4, size/4 + size + size/4 + size, size/4+size)

    content_bbox = np.asarray(content_bbox, dtype=int)
    style_bbox = np.asarray(style_bbox, dtype=int)
    iter_bbox = np.asarray(iter_bbox, dtype=int)

    content_image = Image.open(content_path)
    style_image = Image.open(style_path)

    base_white_background = 255*np.ones(shape=image_size, dtype=np.uint8)
    base_img = Image.fromarray(base_white_background)

    draw_handler_base = ImageDraw.Draw(base_img)
    draw_handler_base.text((iter_bbox[0], iter_bbox[3] + 30), "github: kopytjuk/neural-style-tf-eager", (0, 0, 0), font=font)
    
    iter_image_paths = list()
    for f in os.listdir(iter_path):
        if f.endswith(".png") or f.endswith(".jpg"):
            iter_image_paths.append(os.path.join(iter_path, f))

    iter_image_paths = sorted(iter_image_paths)

    images_list = list()

    for iter_img_path in iter_image_paths:

        i_name = os.path.splitext(os.path.basename(iter_img_path))[0]
        iter_img = Image.open(iter_img_path)

        base_img_i = base_img.copy()

        # paste content, iter and style images
        base_img_i.paste(content_image, content_bbox)
        base_img_i.paste(style_image, style_bbox)
        base_img_i.paste(iter_img, iter_bbox)

        draw_handler = ImageDraw.Draw(base_img_i)
        draw_handler.text((iter_bbox[0], 30), i_name, (0, 0, 0), font=font)
        
        base_img_i = base_img_i.resize(image_size[:2][::-1]//2)
        images_list.append(base_img_i)

    with imageio.get_writer(outpath, mode='I', duration = 0.4) as writer:
        for img in images_list:
            writer.append_data(np.asarray(img))


if __name__=="__main__":

    path = 'test.gif'
    content_path = "img-raw/content2.png"
    style_path = "img-raw/style4.jpg"
    iter_path = "results/20181221-135523/img"

    create_training_gif(path, content_path, style_path, iter_path, size=512)