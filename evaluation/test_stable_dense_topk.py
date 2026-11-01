import unittest

import torch

from .stable_dense_topk import stable_dense_topk


class StableDenseTopKTests(unittest.TestCase):
    def test_boundary_ties_include_omitted_smaller_docids(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            for n, k in [(17, 1), (200, 200), (10000, 200), (1000, 7)]:
                g = torch.Generator().manual_seed(n + k)
                scores = torch.randint(-3, 4, (n,), generator=g).float().to(device)
                order = torch.argsort(scores, descending=True, stable=True)[:k].cpu().tolist()
                self.assertEqual([i for i, _ in stable_dense_topk(scores, k)], order)
                self.assertEqual(stable_dense_topk(scores, k), [(i, float(scores[i])) for i in order])

    def test_all_equal_and_distinct_values(self):
        for scores in [torch.zeros(1001), torch.arange(1001).float().flip(0)]:
            actual = stable_dense_topk(scores, 200)
            expected = torch.argsort(scores, descending=True, stable=True)[:200].tolist()
            self.assertEqual([i for i, _ in actual], expected)

    def test_invalid_k_is_rejected(self):
        for k in (0, -1, 5, 1.5):
            with self.assertRaises(ValueError):
                stable_dense_topk(torch.ones(4), k)


if __name__ == '__main__':
    unittest.main()
