import unittest

import numpy as np
from PIL import Image

from ml.dino_crops import DinoFeatures


class DenseTileTests(unittest.TestCase):
    def test_native_tiles_cover_last_row_and_column(self):
        encoder = DinoFeatures.__new__(DinoFeatures)
        def fake(tile):
            self.assertEqual(tile.size, (224, 224))
            return np.zeros((256, 384))
        encoder.dense_tile = fake
        tiles = list(encoder.tiles(Image.new('RGB', (464, 848))))
        covered = np.zeros((848, 464), bool)
        for _, left, top in tiles:
            covered[top:top+224, left:left+224] = True
        self.assertTrue(covered.all())
        self.assertEqual(max(left for _, left, _ in tiles)+224, 464)
        self.assertEqual(max(top for _, _, top in tiles)+224, 848)

    def test_small_input_is_padded_once_without_stretching(self):
        encoder = DinoFeatures.__new__(DinoFeatures)
        encoder.dense_tile = lambda tile: np.zeros((256, 384))
        self.assertEqual(len(list(encoder.tiles(Image.new('RGB', (100, 120))))), 1)


if __name__ == '__main__':
    unittest.main()
