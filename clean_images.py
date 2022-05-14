from PIL import Image
import os
from os import makedirs
import glob

class CleanImagesFB():
    
    def __init__(self, path: str='./images', target_folder: str='./new_images') -> None:
        self.path = path
        self.target_folder = target_folder
        self.images = glob.glob(self.path + '/*.jpg')
        self.resized = glob.glob(self.target_folder + '/*.jpg')
        makedirs(target_folder, exist_ok=True)
    
    def clean_images(self, size: int=128) -> None:

        final_size = (size, size)
        list_of_cleaned_images = [x.split("/")[-1]
                                  for x in self.resized]
        for image in self.images:
            print("Image:")
            if str(image.split("/")[-1]) not in list_of_cleaned_images:
                print(f'{image.split("/")[-1]} not in new_images')
                print("Cleaning Image")
                image_name = os.path.basename(image)
                print(f'image_name: {image_name}')
                #create black background
                black_img = Image.new(mode='RGB', size=(final_size))
                # open image
                img = Image.open(image)
                print(f'image_name: {image_name}')
                # resize by finding the biggest side of the image and calculating ratio to resize by
                max_dim = max(img.size)
                ratio = final_size[0] / max_dim
                prev_size = img.size
                new_img_size = (int(prev_size[0] * ratio), int(prev_size[1] * ratio))
                img = img.resize(new_img_size)
                #convert to RGB
                img = img.convert('RGB')
                #paste on black background
                black_img.paste(img, (int((final_size[0] - new_img_size[0]) // 2), int((final_size[1] - new_img_size[1]) // 2)))
                #save Images
                print(f'Saving cleaned image: new_images/{image_name}')
                black_img.save(f'new_images/{image_name}')
            else:
                print(f'Image already cleaned: {image}')

if __name__ == '__main__':
    cleaner = CleanImagesFB()
    cleaner.clean_images()
    