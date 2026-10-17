"""Small local CPU visual encoder. Runtime loads only a verified local ONNX file."""
import json
import time

import numpy as np
from PIL import Image

from ml.vision_crops import normalize


def preprocess(image, config):
    image = image.convert('RGB')
    size = config['size']['shortest_edge']
    ratio = size/min(image.size)
    image = image.resize((int(image.width*ratio), int(image.height*ratio)),
                         resample=config.get('resample', Image.Resampling.BICUBIC))
    crop = config['crop_size']
    left, top = (image.width-crop['width'])//2, (image.height-crop['height'])//2
    image = image.crop((left, top, left+crop['width'], top+crop['height']))
    pixels = np.asarray(image, dtype=np.float32)*config.get('rescale_factor', 1/255)
    pixels = (pixels-np.array(config['image_mean'], dtype=np.float32))/np.array(config['image_std'], dtype=np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


class DinoFeatures:
    def __init__(self, path):
        import onnxruntime as ort

        started = time.monotonic()
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path/'model.onnx'), options,
                                            providers=['CPUExecutionProvider'])
        self.config = json.loads((path/'preprocessor_config.json').read_text())
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = next(item.name for item in self.session.get_outputs() if item.name == 'last_hidden_state')
        self.load_sec = time.monotonic()-started

    def encode(self, image):
        hidden = self.session.run([self.output_name],
                                  {self.input_name: preprocess(image, self.config)})[0][0]
        vector = np.concatenate([normalize(hidden[0]), normalize(hidden[1:].mean(axis=0))])
        return normalize(vector).astype(np.float32)

    def dense_tile(self, image):
        """16x16 patch features for a native 224px tile; no positional interpolation."""
        if image.size != (224, 224):
            raise ValueError('Dense tiles must be 224x224')
        pixels = np.asarray(image.convert('RGB'), dtype=np.float32)/255
        pixels = (pixels-np.array(self.config['image_mean'], dtype=np.float32))/np.array(self.config['image_std'], dtype=np.float32)
        hidden = self.session.run([self.output_name],
             {self.input_name: np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])})[0][0]
        if len(hidden) != 257:
            raise ValueError('Dense encoder grid mismatch')
        return normalize(hidden[1:]).astype(np.float32)

    def tiles(self, image):
        """Cover the full image at native resolution, including its final edges."""
        image = image.convert('RGB')
        def starts(length):
            return sorted(set([*range(0, max(1, length-223), 112), max(0, length-224)]))
        for top in starts(image.height):
            for left in starts(image.width):
                yield self.dense_tile(image.crop((left, top, left+224, top+224))), left, top
