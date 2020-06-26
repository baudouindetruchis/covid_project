from PIL import Image
import math
from matplotlib import pyplot as plt


def image_resize(image, target_size=(100,100)):
    """Resize an image while keeping it aspect ratio --> add padding if necessary"""
    # Resize with same aspect ratio
    old_width, old_height = image.shape[1], image.shape[0]
    ratio = min(target_size[0]/old_width, target_size[1]/old_height)
    new_width, new_height = math.floor(ratio*old_width), math.floor(ratio*old_height)
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((new_width, new_height))

    # Add padding
    background = Image.new('RGB', (target_size[0], target_size[1]))
    paste_x = math.floor((target_size[0] - new_width)/2)
    paste_y = math.floor((target_size[1] - new_height)/2)
    background.paste(resized, (paste_x, paste_y))

    return background


# ========== RUNING ==========

if __name__ == "__main__":
    project_path = 'D:/code#/[large_data]/covid_project/'
    location = 'serbia'

    yolo_folder = project_path + 'models/yolo_coco/'
    image_folder = project_path + 'video_scraping/' + location + '/'

    frame = plt.imread(image_folder + 'serbia_1592484700624.jpg')
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(frame)
    plt.subplot(1,2,2)
    plt.imshow(image_resize(frame, target_size=(700,300)))
    plt.show()
