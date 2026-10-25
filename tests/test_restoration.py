import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from model import restore_diacritics


class PositionScores(torch.nn.Module):
    """Supply known aligned scores to exercise the real restoration code."""

    def __init__(self, dataset, target):
        super().__init__()
        self.dataset = dataset
        self.target = target
        self.calls = 0

    def forward(self, src):
        self.calls += 1
        scores = torch.zeros(1, src.size(1), len(self.dataset.char_to_idx))
        # An illegal candidate wins globally, testing constrained selection too.
        scores[:, :, self.dataset.bos_idx] = 100
        for position, char in enumerate(self.target, start=1):
            scores[0, position, self.dataset.char_to_idx[char]] = 10
        return scores


def notebook_restorer():
    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / 'inference_kaggle.ipynb').read_text()
    )
    for cell in notebook['cells']:
        source = ''.join(cell.get('source', []))
        if cell['cell_type'] == 'code' and 'def restore_diacritics(' in source:
            tree = ast.parse(source)
            function = next(node for node in tree.body
                            if isinstance(node, ast.FunctionDef)
                            and node.name == 'restore_diacritics')
            namespace = {'torch': torch, 'nn': torch.nn,
                         'MinimalDataset': SimpleNamespace}
            exec(compile(ast.Module(body=[function], type_ignores=[]),
                         'inference_kaggle.ipynb', 'exec'), namespace)
            return namespace['restore_diacritics']
    raise AssertionError('Notebook restoration function not found')


class RestorationTest(unittest.TestCase):
    def test_scores_align_with_characters_after_bos(self):
        tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] + list('оу нөүңа!')
        char_to_idx = {char: i for i, char in enumerate(tokens)}
        dataset = SimpleNamespace(
            char_to_idx=char_to_idx,
            idx_to_char={i: char for char, i in char_to_idx.items()},
            bos_idx=2, eos_idx=3, unk_idx=1,
        )
        for source, restore in [('model.py', restore_diacritics),
                                ('inference_kaggle.ipynb', notebook_restorer())]:
            for text, expected in [('оо', 'өо'), ('уу', 'үу'), ('нн', 'ңн'),
                                   ('аоу н!', 'аөү ң!'), ('', ''), ('а!', 'а!')]:
                with self.subTest(source=source, text=text):
                    model = PositionScores(dataset, expected)
                    self.assertEqual(restore(model, text, dataset, torch.device('cpu')),
                                     expected)
                    self.assertEqual(model.calls, 1)


if __name__ == '__main__':
    unittest.main()
