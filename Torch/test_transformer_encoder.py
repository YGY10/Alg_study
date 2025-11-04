import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import unittest

class TestSimpleSelfAttention(unittest.TestCase):
    """
    测试 SimpleSelfAttention 类的功能，包括正常输入、边界条件和异常输入。
    """

    def setUp(self):
        """
        初始化测试环境，创建 SimpleSelfAttention 实例。
        """
        self.embed_size = 32
        self.heads = 4
        self.model = SimpleSelfAttention(self.embed_size, self.heads)

    def test_forward_normal_input(self):
        """
        测试正常输入下的 forward 方法。
        """
        # 输入形状: (N, seq_len, embed_size)
        x = torch.randn(2, 10, self.embed_size)
        output = self.model(x)
        self.assertEqual(output.shape, (2, 10, self.embed_size))

    def test_forward_edge_case_empty_sequence(self):
        """
        测试空序列输入下的 forward 方法。
        """
        # 输入形状: (N, 0, embed_size)
        x = torch.randn(2, 0, self.embed_size)
        output = self.model(x)
        self.assertEqual(output.shape, (2, 0, self.embed_size))

    def test_forward_edge_case_single_token(self):
        """
        测试单 token 输入下的 forward 方法。
        """
        # 输入形状: (N, 1, embed_size)
        x = torch.randn(2, 1, self.embed_size)
        output = self.model(x)
        self.assertEqual(output.shape, (2, 1, self.embed_size))

    def test_forward_invalid_input_shape(self):
        """
        测试无效输入形状下的 forward 方法。
        """
        # 输入形状: (N, seq_len, wrong_embed_size)
        x = torch.randn(2, 10, self.embed_size + 1)
        with self.assertRaises(RuntimeError):
            self.model(x)


class TestSimpleTransformerBlock(unittest.TestCase):
    """
    测试 SimpleTransformerBlock 类的功能，包括正常输入和边界条件。
    """

    def setUp(self):
        """
        初始化测试环境，创建 SimpleTransformerBlock 实例。
        """
        self.embed_size = 32
        self.heads = 4
        self.model = SimpleTransformerBlock(self.embed_size, self.heads)

    def test_forward_normal_input(self):
        """
        测试正常输入下的 forward 方法。
        """
        # 输入形状: (N, seq_len, embed_size)
        x = torch.randn(2, 10, self.embed_size)
        output = self.model(x)
        self.assertEqual(output.shape, (2, 10, self.embed_size))

    def test_forward_edge_case_empty_sequence(self):
        """
        测试空序列输入下的 forward 方法。
        """
        # 输入形状: (N, 0, embed_size)
        x = torch.randn(2, 0, self.embed_size)
        output = self.model(x)
        self.assertEqual(output.shape, (2, 0, self.embed_size))


class TestLightPositionalEncoding(unittest.TestCase):
    """
    测试 LightPositionalEncoding 类的功能，包括正常输入和边界条件。
    """

    def setUp(self):
        """
        初始化测试环境，创建 LightPositionalEncoding 实例。
        """
        self.embed_size = 32
        self.model = LightPositionalEncoding(self.embed_size)

    def test_forward_normal_input(self):
        """
        测试正常输入下的 forward 方法。
        """
        # 输入形状: (N, seq_len, embed_size)
        x = torch.randn(2, 10, self.embed_size)
        output = self.model(x)
        self.assertEqual(output.shape, (2, 10, self.embed_size))

    def test_forward_edge_case_empty_sequence(self):
        """
        测试空序列输入下的 forward 方法。
        """
        # 输入形状: (N, 0, embed_size)
        x = torch.randn(2, 0, self.embed_size)
        output = self.model(x)
        self.assertEqual(output.shape, (2, 0, self.embed_size))


if __name__ == "__main__":
    unittest.main()