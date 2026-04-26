import base64
import io
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
import ast
from openai import OpenAI


class CostMap:
    def __init__(
        self, name, image_sets, cost_dict=None):
        self.name = name
        self.image_sets = image_sets
        self.cost_dict = cost_dict
        self.points = self._generate_points()
        self.voxels = self._generate_voxels()
        self.voxel_grid = self._generate_voxel_grid()


    def _generate_points(self):
        print("Generating points...")
        total_points = []
        for image_set in self.image_sets:
            camera = image_set.camera
            near = camera.z_range[0]
            far = camera.z_range[1]
            fx = camera.intrinsics[0, 0]
            fy = camera.intrinsics[1, 1]
            cx = camera.intrinsics[0, 2]
            cy = camera.intrinsics[1, 2]

            depth_buffer = image_set.depth * 2.0 - 1.0
            depth = (2.0 * near * far) / (far + near - depth_buffer * (far - near))

            h, w = depth.shape
            xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))

            x = (xmap - cx) * depth / fx
            y = (ymap - cy) * depth / fy
            z = depth

            points_cam = np.float32([x, y, z]).transpose(1, 2, 0)
            transform = np.eye(4)
            transform[:3, :] = np.hstack((camera.rotation, np.array(camera.position).reshape(3, 1)))
            points_world = transform_pointcloud(points_cam, transform)
            points_world = points_world.reshape(-1, 3)

            mask = image_set.label.flatten() > 0
            labels = image_set.label.flatten()[mask]
            cost_function = np.vectorize(self.cost_dict.get)
            points_world = points_world[mask, :]
            colors = image_set.color.reshape(-1, 3)[mask]
            costs = cost_function(labels)

            points = np.concatenate((
                points_world,
                colors,
                labels.reshape(-1, 1),
                costs.reshape(-1, 1),
            ), axis=-1)
            total_points.append(points)

        total_points = np.concatenate(total_points, axis=0)
        print("Points completed.")
        return total_points
    

    def _generate_voxels(self):
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]
        costs = self.points[:, -1]

        D_x, D_y, D_z = (100, 100, 100)
        x_min, x_max = -0.5, 0.5
        y_min, y_max = -0.5, 0.5
        z_min, z_max = 0, 1

        range_x = x_max - x_min
        range_y = y_max - y_min
        range_z = z_max - z_min

        voxel_size_x = range_x / D_x
        voxel_size_y = range_y / D_y
        voxel_size_z = range_z / D_z

        self.voxel_size = np.array([voxel_size_x, voxel_size_y, voxel_size_z])
        self.origin = np.array([x_min, y_min, z_min])

        x -= x_min
        y -= y_min
        z -= z_min

        x_voxel = (x / voxel_size_x).astype(int)
        y_voxel = (y / voxel_size_y).astype(int)
        z_voxel = (z / voxel_size_z).astype(int)

        x_voxel = np.clip(x_voxel, 0, D_x - 1)
        y_voxel = np.clip(y_voxel, 0, D_y - 1)
        z_voxel = np.clip(z_voxel, 0, D_z - 1)

        voxel_indices = np.stack((x_voxel, y_voxel, z_voxel), axis=1)

        unique_voxels, unique_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

        average_costs = []

        for unique_voxel_index in tqdm(range(len(unique_voxels)), desc="Processing Voxel Grid"):
            unique_voxel_mask = (unique_indices == unique_voxel_index).flatten()
            unique_voxel_costs = costs[unique_voxel_mask]
            unique_voxel_costs = [cost for cost in unique_voxel_costs if cost is not None]

            if unique_voxel_costs:
                average_cost = np.mean(unique_voxel_costs)
            else:
                average_cost = 0

            average_costs.append(average_cost)

        average_costs = np.array(average_costs).astype(float)
        voxels = np.column_stack((unique_voxels, average_costs))
        return voxels


    def _generate_voxel_grid(self):
        x_dim, y_dim, z_dim = 100, 100, 100

        voxel_grid = np.zeros((x_dim, y_dim, z_dim), dtype=np.float32)

        for x, y, z, c in self.voxels:
            x, y, z = int(x) - 1, int(y) - 1, int(z) - 1
            if x < 0 or x >= x_dim or y < 0 or y >= y_dim or z < 0 or z >= z_dim:
                continue
            voxel_grid[x, y, z] = c

        return voxel_grid
    

    def visualize_voxel_grid(self):
        x = self.voxels[:, 0]
        y = self.voxels[:, 1]
        z = self.voxels[:, 2]
        costs = self.voxels[:, -1].astype(float)

        colormap = plt.cm.viridis
        norm = mpl.colors.Normalize(vmin=0, vmax=10)
        np.set_printoptions(threshold=np.inf)
        colors = (colormap(norm(costs))[:, :3] * 255).astype(int)
        colors = ['rgb({},{},{})'.format(r, g, b) for r, g, b in colors]

        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=3,
                color=colors,
                opacity=1,
            ),
            text=costs,
        )

        fig = go.Figure(data=[trace])

        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.4)
            ),
            showlegend=False,
            margin=dict(
                l=0,  
                r=0, 
                b=0, 
                t=0   
            )
        )
        fig.show()
    
    
def transform_pointcloud(points, transform):
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding,'constant', constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points
    

def create_prompt(image_set):
    object_list = "".join(
        f"- ({i+2}) " + scene_object_name + "\n"
        for i, scene_object_name in image_set.scene_objects.items()
    )

    prompt = f"""
You are an advanced AI system designed to assist in robotic navigation within cluttered environments. 
Your task is to evaluate the safety of various objects in a scene, considering not only the individual characteristics of each object but also their interactions and the physical principles that govern their stability and potential movement. 
For each listed object, you will assign a safety score ranging from 0 to 10, where 0 indicates minimal risk of damage upon collision and 10 signifies a high risk of damage. 
Your assessment should consider the object's material, size, context, and its relationship with surrounding objects, including any potential domino effects or other interaction-induced risks.

Please provide a safety score for each object, taking into account the following:
- Object Material and Structural Integrity: How fragile or sturdy the object is (e.g., glass vs. plastic vs. metal).
- Stability and Likelihood of Movement: Whether the object is firmly placed or precariously balanced. For instance, a stable stack of books alone might be low risk, but if there is a fragile object on top, the risk increases significantly.
- Potential Domino Effects or Chain Reactions: If collision with one object could cause it to roll, topple, or otherwise move into other objects, increasing the overall risk. For example, a ball might be low risk in isolation, but if it can roll and knock over a wine glass, the effective risk is higher.
- Proximity and Arrangement: How close the object is to other fragile or easily toppled items. Even if an object (like a sugar box) is normally sturdy, being positioned next to a fragile wine glass can raise its overall risk score if it could collide or push the glass.
- Any Other Relevant Physical Interactions: Any additional factors that might increase the risk of damage, such as height above the ground, shape of the surface, or presence of liquids.

Each item is an object labelled in white with its respective ID number. Adhere to the specified format for your response, listing each object followed by its corresponding safety score. Do not include any additional text or output.

Format Requirements:
- The JSON object must be a single string.
- Each key must be the object's ID number in parentheses (e.g., "1"), and each value must be the safety score (an integer between 0 and 10).
- Do not include any text other than the JSON object in that final line. Do not add something like "```json" or "```".
- Do not include newlines, extra punctuation, or object names in the JSON.
- Every key/value should be strictly "ID": score.
- No explanations or reasoning should appear in the final JSON—only the scores.

Input objects:
{object_list}
Your analysis should be comprehensive, considering the dynamic interactions between objects and the physical principles that may affect the outcome of a collision.
"""
    return prompt


def fetch_cost_dict_bypass(image_set):
    # Mock: assign a neutral safety score of 5 to all objects (bypasses OpenAI API)
    cost_dict = {i + 2: 3+i*2 for i in image_set.scene_objects}
    print(f"Mock cost_dict: {cost_dict}")
    return cost_dict


def fetch_cost_dict(image_set):
    model = "gpt-4o" 
    buffered = io.BytesIO()
    image_set.annotated_image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    prompt = create_prompt(image_set)

    payload = { 
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 800,
    }

    client = OpenAI()   

    completion = client.chat.completions.create(model=model, messages=payload["messages"])

    content = completion.choices[0].message.content
    print(f"{model} Output: {content}")

    content_dict = ast.literal_eval(content)
    try:
        cost_dict = {int(i): cost for i, cost in content_dict.items()}
    except:
        cost_dict = {int(i[1:-1]): cost for i, cost in content_dict.items()}
    return cost_dict
    