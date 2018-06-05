import unittest

from random import randint

import dataset_builder as db

"""
LABELS SUMMARY

PVC_LABEL: 1
JOINT_LABEL: 2
GLASS_LABEL: 3
WOOD_LABEL: 4
PE_LABEL: 5
PA_LABEL: 5
PS_LABEL: 5
"""

def gen_path(t):
    """
        Given a material type, return a fake absolute path to the image

        Args:
            t -- string -- the material's type

        return a string, the image's absolute path
    """
    fake_path = "/home/yolo/Documents/"
    return fake_path + t + "_" + str(randint(1, 10000)) + ".ppm"

class TestLabelize(unittest.TestCase):

    def test_pvc_vs_all(self):
        """ Test that the function puts the right labels for pvc vs all
        """
        d = db.pvc_vs_all()
        self.assertEqual(db.labelize(gen_path("PVC"), d), 1)
        self.assertEqual(db.labelize(gen_path("Pvc"), d), 1)
        self.assertEqual(db.labelize(gen_path("Joint"), d), 0)
        self.assertEqual(db.labelize(gen_path("Verre"), d), 0)
        self.assertEqual(db.labelize(gen_path("Bois"), d), 0)
        self.assertEqual(db.labelize(gen_path("PE"), d), 0)
        self.assertEqual(db.labelize(gen_path("PA"), d), 0)
        self.assertEqual(db.labelize(gen_path("PS"), d), 0)

    def test_existing_uncorrect_file(self):
        """ Test that the function puts the right labels for pvc vs wood vs glass
            vs joint
        """
        d = db.pvc_joint_glass_wood_other()
        self.assertEqual(db.labelize(gen_path("PVC"), d), 1)
        self.assertEqual(db.labelize(gen_path("Pvc"), d), 1)
        self.assertEqual(db.labelize(gen_path("JoINT"), d), 2)
        self.assertEqual(db.labelize(gen_path("Verre"), d), 3)
        self.assertEqual(db.labelize(gen_path("VeRRe"), d), 3)
        self.assertEqual(db.labelize(gen_path("BOIS"), d), 4)
        self.assertEqual(db.labelize(gen_path("bois"), d), 4)
        self.assertEqual(db.labelize(gen_path("PE"), d), 0)
        self.assertEqual(db.labelize(gen_path("PA"), d), 0)
        self.assertEqual(db.labelize(gen_path("PS"), d), 0)

    def test_unexisting_file(self):
        """ Test that the function puts the right labels for pvc vs wood vs glass
            vs joint vs PE/PA/PS
        """
        d = db.pvc_joint_glass_wood_pePaPs_other()
        self.assertEqual(db.labelize(gen_path("PVC"), d), 1)
        self.assertEqual(db.labelize(gen_path("Pvc"), d), 1)
        self.assertEqual(db.labelize(gen_path("JoINT"), d), 2)
        self.assertEqual(db.labelize(gen_path("Verre"), d), 3)
        self.assertEqual(db.labelize(gen_path("VeRRe"), d), 3)
        self.assertEqual(db.labelize(gen_path("BOIS"), d), 4)
        self.assertEqual(db.labelize(gen_path("bois"), d), 4)
        self.assertEqual(db.labelize(gen_path("PE"), d), 5)
        self.assertEqual(db.labelize(gen_path("pE"), d), 5)
        self.assertEqual(db.labelize(gen_path("PA"), d), 5)
        self.assertEqual(db.labelize(gen_path("Pa"), d), 5)
        self.assertEqual(db.labelize(gen_path("PS"), d), 5)
        self.assertEqual(db.labelize(gen_path("ps"), d), 5)

if __name__ == "__main__":
    unittest.main()
