import unittest
from process.data_processor_type import *

class TestKeyMatch(unittest.TestCase):

    # 首先测试name匹配
    def test_TDP_name_match(self):
        # both cnt = 1
        #   same type
        po = [('List[str]', 'a')]
        pi = [('List[str]', 'a')]
        self.assertEqual(key_match(po, pi), {('List[str]', 'a'): ('List[str]', 'a')})
        #   subtype
        po = [('List[str]', 'a')]
        pi = [('Sequence[str]', 'a')]
        self.assertEqual(key_match(po, pi), {('List[str]', 'a'): ('Sequence[str]', 'a')})
        #   different type
        po = [('List[str]', 'a')]
        pi = [('int', 'a')]
        self.assertEqual(key_match(po, pi), {('List[str]', 'a'): ('int', 'a')})

        # both cnt = 2
        #   same type 1
        po = [('List[str]', 'a'), ('List[str]', 'b')]
        pi = [('List[str]', 'a'), ('List[str]', 'b')]
        self.assertEqual(key_match(po, pi), {
            ('List[str]', 'a'): ('List[str]', 'a'),
            ('List[str]', 'b'): ('List[str]', 'b')})
        #   same type 2
        po = [('List[str]', 'a'), ('List[int]', 'b')]
        pi = [('List[str]', 'a'), ('List[int]', 'b')]
        self.assertEqual(key_match(po, pi), {
            ('List[str]', 'a'): ('List[str]', 'a'),
            ('List[int]', 'b'): ('List[int]', 'b')})
        #   subtype
        po = [('List[str]', 'a'), ('List[str]', 'b')]
        pi = [('Sequence[str]', 'a'), ('Iterable[str]', 'b')]
        self.assertEqual(key_match(po, pi), {
            ('List[str]', 'a'): ('Sequence[str]', 'a'),
            ('List[str]', 'b'): ('Iterable[str]', 'b')})
        #   different type
        po = [('List[str]', 'a'), ('List[int]', 'b')]
        pi = [('Dict[str, str]', 'a'), ('float', 'b')]
        self.assertEqual(key_match(po, pi), {
            ('List[str]', 'a'): ('Dict[str, str]', 'a'),
            ('List[int]', 'b'): ('float', 'b')})

        # both cnt > 2
        po = [('str', 'a'), ('int', 'b'), ('float', 'c')]
        pi = [('str', 'b'), ('float', 'c')]
        self.assertEqual(key_match(po, pi), {
            ('int', 'b'): ('str', 'b'),
            ('float', 'c'): ('float', 'c')
        })

    def test_TDP_equal_type_match(self):
        # both cnt = 1
        #   no match
        po = [('List[str]', 'a')]
        pi = [('List[int]', 'd')]
        self.assertEqual(key_match(po, pi), {('List[str]', 'a'): ('List[int]', 'd')})
        #   match
        po = [('List[str]', 'a')]
        pi = [('List[str]', 'd')]
        self.assertEqual(key_match(po, pi), {
            ('List[str]', 'a'): ('List[str]', 'd')
        })

        # both cnt = 2
        #   full match
        po = [('List[str]', 'a'), ('List[int]', 'b')]
        pi = [('List[int]', 'c'), ('List[str]', 'd')]
        self.assertEqual(key_match(po, pi), {
            ('List[str]', 'a'): ('List[str]', 'd'),
            ('List[int]', 'b'): ('List[int]', 'c')})
        #   part match
        po = [('List[str]', 'a'), ('List[float]', 'b')]
        pi = [('List[int]', 'c'), ('List[str]', 'd')]
        self.assertEqual(key_match(po, pi), {
            ('List[str]', 'a'): ('List[str]', 'd')})
        #   no match
        po = [('List[str]', 'a'), ('List[float]', 'b')]
        pi = [('List[int]', 'c'), ('str', 'd')]
        self.assertEqual(key_match(po, pi), {})

    def test_TDP_subtype_match(self):
        pass

    def test_TDP_cnt_match(self):
        pass

    def test_TDP_name_spec(self):
        """
        测试key_match是否正确检查了的name的spec
        :return:
        """

    def test_TDP_type_spec(self):
        """
        测试key_match是否正确检查了type的spec
        :return:
        """




if __name__ == '__main__':
    unittest.main()
