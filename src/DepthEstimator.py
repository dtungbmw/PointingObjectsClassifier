from transformers import GLPNImageProcessor, GLPNForDepthEstimation
from PIL import Image
import torch
import timm
import torchvision.transforms as T
#from depth_anything_v2.dpt import DepthAnythingV2


class DepthEstimator:
    
    def predict(self):
        pass

'''
class AnythingDepthEstimator(DepthEstimator):
    def __init__(self):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        encoder = 'vitb'  # or 'vits', 'vitb', 'vitg'
        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.model = self.model.to(DEVICE).eval()

    def predict(self, image):
        depth = self.model.infer_image(image)
        return depth
'''

class AdabinsDepthEstimator(DepthEstimator):

    def __init__(self):
        self.model = timm.create_model('adabins', pretrained=True)
        self.model.eval()

    def predict(self, image):
        with torch.no_grad():
            depth_map = self.model(image)
        depth_map = depth_map.squeeze().cpu().numpy()
        return depth_map

    
class GLPNDepthEstimator(DepthEstimator):

    def __init__(self):
        self.feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
        self.model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

    def predict(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            return predicted_depth


class MidasDepthEstimator(DepthEstimator):

    def __init__(self):
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.midas.eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        #model_type = "DPT_Hybrid"
        #model_path = "model-f6b98070.pt"
        #midas_model = load_model(model_path, model_type)

    def predict(self, img):
        input_batch = self.transform(img).to("cuda" if torch.cuda.is_available() else "cpu")

        # Perform inference to get depth map
        with torch.no_grad():
            prediction = midas(input_batch)

        # Upsample the prediction to the original image size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Convert depth to numpy
        depth_map = prediction.cpu().numpy()
        return depth_map


    