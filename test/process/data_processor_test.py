import unittest


from process.data_processor import *

class TestProcessorManager(unittest.TestCase):

    def test_DP_add1(self):
        dp_A_B = DataProcessor('A', 'B')
        dp_B_C = DataProcessor('B', 'C')
        dp_result = dp_A_B + dp_B_C
        self.assertEqual(dp_result.input_keys, tuple(['A']))
        self.assertEqual(dp_result.output_keys, tuple(['C']))

    def test_DP_add2(self):
        dp_A_B = DataProcessor('A', 'B')
        dp_C_D = DataProcessor('C', 'D')
        dp_result = dp_A_B + dp_C_D
        self.assertEqual(set(dp_result.input_keys), {'A', 'C'})
        self.assertEqual(set(dp_result.output_keys), {'B', 'D'})

    def test_DP_add3(self):
        dp_A_BC = DataProcessor('A', ['B', 'C'])
        dp_CD_E = DataProcessor(['C', 'D'], 'E')
        dp_result = dp_A_BC + dp_CD_E
        self.assertEqual(set(dp_result.input_keys), {'A', 'D'})
        self.assertEqual(set(dp_result.output_keys), {'B', 'E'})

    def test_DP_add4(self):
        dp_A_B = DataProcessor('A', 'B')
        dp_B_C = DataProcessor('B', 'C')
        dp_C_D = DataProcessor('C', 'D')
        dp_D_E = DataProcessor('D', 'E')
        dp_result = dp_A_B + dp_B_C + dp_C_D + dp_D_E
        self.assertEqual(dp_result.input_keys, tuple(['A']))
        self.assertEqual(dp_result.output_keys, tuple(['E']))

    def test_DP_add5(self):
        dp_A_B = DataProcessor('A', 'B')
        dp_C_D = DataProcessor('C', 'D')
        dp_E_F = DataProcessor('E', 'F')
        dp_result = dp_A_B + dp_C_D + dp_E_F
        self.assertEqual(set(dp_result.input_keys), {'A', 'C', 'E'})
        self.assertEqual(set(dp_result.output_keys), {'B', 'D', 'F'})

    def test_DP_add6(self):
        dp_A_B = DataProcessor('A', 'B')
        dp_B_C = DataProcessor('B', 'C')
        dp_D_E = DataProcessor('D', 'E')
        dp_F_G = DataProcessor('F', 'G')
        dp_result = dp_A_B + dp_B_C + dp_D_E + dp_F_G
        self.assertEqual(set(dp_result.input_keys), {'A', 'D', 'F'})
        self.assertEqual(set(dp_result.output_keys), {'C', 'E', 'G'})

    def test_DP_mul1(self):
        dp_A_B = DataProcessor('A', 'B')
        dp_C_D = DataProcessor('C', 'D')
        dp_result = dp_A_B * dp_C_D
        self.assertEqual(set(dp_result.input_keys), {'A', 'C'})
        self.assertEqual(set(dp_result.output_keys), {'B', 'D'})

    def test_DP_mul2(self):
        dp_A_B = DataProcessor('A', 'B')
        dp_C_D = DataProcessor('C', 'D')
        dp_E_F = DataProcessor('E', 'F')
        dp_result = dp_A_B * dp_C_D * dp_E_F
        self.assertEqual(set(dp_result.input_keys), {'A', 'C', 'E'})
        self.assertEqual(set(dp_result.output_keys), {'B', 'D', 'F'})

    def test_PM_add1(self):
        dp_A_B = DataProcessor('A', 'B')
        dp_C_D = DataProcessor('C', 'D')
        dp_BD_E = DataProcessor(['B', 'D'], 'E')
        PM1 = dp_A_B * dp_C_D
        PM_result = PM1 + dp_BD_E
        self.assertEqual(set(PM_result.input_keys), {'A', 'C'})
        self.assertEqual(set(PM_result.output_keys), {'E'})

    def test_PM_add2(self):
        dp_A_B = DataProcessor('A', 'B')
        dp_C_D = DataProcessor('C', 'D')
        dp_B_E = DataProcessor('B', 'E')
        PM1 = dp_A_B * dp_C_D
        PM_result = PM1 + dp_B_E
        self.assertEqual(set(PM_result.input_keys), {'A', 'C'})
        self.assertEqual(set(PM_result.output_keys), {'D', 'E'})

if __name__ == '__main__':
    unittest.main()
