import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.image as mpimg

def heatmap(descriptors, path_output, file_name):
    plt.figure(figsize = (20,20))
    sns.heatmap((descriptors).select_dtypes(include=['number']).corr(), annot = True, vmin=-1, vmax=1, center= 0)
    plt.title("heatmap признаков", fontsize=20)

    output_path = os.path.join(path_output, file_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def display_images(folder_path):
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = mpimg.imread(img_path)
        plt.figure(figsize=(8, 8)) 
        plt.imshow(img)
        plt.axis('off')
        plt.title(filename)
        plt.show()