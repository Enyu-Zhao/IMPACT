import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ImageSet:
    def __init__(self, scene, arrays, camera):
        self.color, self.depth, self.label = arrays
        self.depth = np.array(self.depth).reshape(camera.image_size)
        self.label = np.array(self.label).reshape(camera.image_size)
        self.color = np.array(self.color).reshape(camera.image_size + (4,)).astype(np.uint8)
        self.color = self.color[:, :, :3]

        self.camera = camera

        self.annotated_image = self._generate_annotated_image()
        self.scene_objects = {
            i: scene_object.name for i, scene_object in enumerate(scene.scene_objects)
        }


    def show(self, type="cdl"):
        if type == "cdl":
            fig, axs = plt.subplots(1, 3)

            for ax, image in zip(axs, [self.color, self.depth, self.label]):
                ax.imshow(image)
        elif type == "annotated":
            plt.imshow(self.annotated_image)
        else:
            image_type = {"color": self.color, "depth": self.depth, "label": self.label}

            plt.imshow(image_type[type])
            plt.colorbar()

        plt.show()


    def _generate_annotated_image(self):
        unique_labels = np.unique(self.label)

        annotated_image = Image.fromarray(self.color)

        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.load_default()

        for label in unique_labels:
            if label == 0:
                continue

            mask = self.label == label
            indices = np.argwhere(mask)
            average_indices = indices.mean(axis=0)
            y, x = average_indices

            draw.text((x, y), str(label), font=font, fill=(255, 255, 255))

        return annotated_image