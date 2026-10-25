import unittest
from run import VARIANTS, build, data


class Protocol(unittest.TestCase):
    def test_shared_parameter_and_compute_controls(self):
        params={k:sum(p.numel() for p in build(*v).parameters()) for k,v in VARIANTS.items()}
        self.assertEqual(params["shared-1"],params["shared-4"])
        self.assertGreater(params["unshared-4"],params["shared-4"])
        self.assertEqual(VARIANTS["shared-4"][0],VARIANTS["unshared-4"][0])

    def test_disjoint_holdout(self):
        _,_,train,test=data()
        self.assertFalse(set(train)&set(test))
