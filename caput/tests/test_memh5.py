"""Unit tests for the memh5 module."""

import unittest
import os
import glob

import numpy as np
import h5py

from caput import memh5

class TestRODict(unittest.TestCase):
    """Unit tests for ro_dict."""

    def test_everything(self):
        a = {'a' : 5}
        a = memh5.ro_dict(a)
        self.assertEqual(a['a'], 5)
        self.assertEqual(list(a.keys()), ['a'])
        # Convoluded test to make sure you can't write to it.
        try: a['b'] = 6
        except TypeError: correct = True
        else: correct = False
        self.assertTrue(correct)


class TestGroup(unittest.TestCase):
    """Unit tests for MemGroup."""

    def test_nested(self):
        root = memh5.MemGroup()
        l1 = root.create_group('level1')
        l2 = l1.require_group('level2')
        self.assertTrue(root['level1'] == l1)
        self.assertTrue(root['level1/level2'] == l2)
        self.assertEqual(root['level1/level2'].name, '/level1/level2')

    def test_create_dataset(self):
        g = memh5.MemGroup()
        data = np.arange(100, dtype=np.float32)
        g.create_dataset('data', data=data)
        self.assertTrue(np.allclose(data, g['data']))

    def test_recursive_create(self):
        g = memh5.MemGroup()
        self.assertRaises(ValueError, g.create_group, '')
        g2 = g.create_group('level2/')
        self.assertRaises(ValueError, g2.create_group, '/')
        g2.create_group('/level22')
        self.assertEqual(set(g.keys()), {'level22', 'level2'})
        g.create_group('/a/b/c/d/')
        gd = g['/a/b/c/d/']
        self.assertEqual(gd.name, '/a/b/c/d')

    def test_recursive_create_dataset(self):
        g = memh5.MemGroup()
        data = np.arange(10)
        g.create_dataset('a/ra', data=data)
        self.assertTrue(memh5.is_group(g['a']))
        self.assertTrue(np.all(g['a/ra'][:] == data))
        g['a'].create_dataset('/ra', data=data)
        print(g.keys())
        self.assertTrue(np.all(g['ra'][:] == data))


class TestH5Files(unittest.TestCase):
    """Tests that make hdf5 objects, convert to mem and back."""

    fname = 'tmp_test_memh5.h5'

    def setUp(self):
        with h5py.File(self.fname, 'w') as f:
            l1 = f.create_group('level1')
            l2 = l1.create_group('level2')
            d1 = l1.create_dataset('large', data=np.arange(100))
            f.attrs['a'] = 5
            d1.attrs['b'] = 6
            l2.attrs['small'] = np.arange(3)

    def assertGroupsEqual(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        self.assertAttrsEqual(a.attrs, b.attrs)
        for key in a.keys():
            this_a = a[key]
            this_b = b[key]
            if not memh5.is_group(a[key]):
                self.assertAttrsEqual(this_a.attrs, this_b.attrs)
                self.assertTrue(np.allclose(this_a, this_b))
            else:
                self.assertGroupsEqual(this_a, this_b)

    def assertAttrsEqual(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for key in a.keys():
            this_a = a[key]
            this_b = b[key]
            if hasattr(this_a, 'shape'):
                self.assertTrue(np.allclose(this_a, this_b))
            else:
                self.assertEqual(this_a, this_b)

    def test_h5_sanity(self):
        f = h5py.File(self.fname, 'r')
        self.assertGroupsEqual(f, f)
        f.close()

    def test_to_from_hdf5(self):
        m = memh5.MemGroup.from_hdf5(self.fname)

        # Check that read in file has same structure
        with h5py.File(self.fname, 'r') as f:
            self.assertGroupsEqual(f, m)

        m.to_hdf5(self.fname + '.new')

        # Check that written file has same structure
        with h5py.File(self.fname + '.new', 'r') as f:
            self.assertGroupsEqual(f, m)

    def test_memdisk(self):
        f = memh5.MemDiskGroup(self.fname)
        self.assertEqual(set(f.keys()), set(f._data.keys()))
        m = memh5.MemDiskGroup(memh5.MemGroup.from_hdf5(self.fname))
        self.assertEqual(set(m.keys()), set(f.keys()))
        # Recursive indexing.
        self.assertEqual(set(f['/level1/'].keys()), set(m['/level1/'].keys()))
        self.assertEqual(set(f.keys()), set(m['/level1']['/'].keys()))
        self.assertTrue(np.all(f['/level1/large'][:] == m['/level1/large']))
        gf = f.create_group('/level1/level2/level3/')
        df = gf.create_dataset('new', data=np.arange(5))
        gm = m.create_group('/level1/level2/level3/')
        dm = gm.create_dataset('new', data=np.arange(5))
        self.assertTrue(np.all(f['/level1/level2/level3/new'][:]
                               == m['/level1/level2/level3/new'][:]))

    def tearDown(self):
        file_names = glob.glob(self.fname + '*')
        for fname in file_names:
            os.remove(fname)


class TempSubClass(memh5.MemDiskGroup):
    pass


class TestMemDiskGroup(unittest.TestCase):

    fname = 'temp_mdg.h5'

    def test_io(self):

        # Save a subclass of MemDiskGroup
        tsc = TempSubClass()
        tsc.create_dataset('dset', data=np.arange(10))
        tsc.save('temp_mdg.h5')

        # Load it from disk
        tsc2 = memh5.MemDiskGroup.from_file('temp_mdg.h5')

        # Check that is is recreated with the correct type
        self.assertIsInstance(tsc2, TempSubClass)

    def tearDown(self):
        file_names = glob.glob(self.fname + '*')
        for fname in file_names:
            os.remove(fname)


class TestBasicCont(unittest.TestCase):

    def test_access(self):
        d = memh5.BasicCont()
        self.assertTrue('history' in d._data.keys())
        self.assertTrue('index_map' in d._data.keys())
        self.assertRaises(KeyError, d.__getitem__, 'history')
        self.assertRaises(KeyError, d.__getitem__, 'index_map')

        self.assertRaises(ValueError, d.create_group, 'a')
        self.assertRaises(ValueError, d.create_dataset, 'index_map/stuff',
                          data=np.arange(5))
        # But make sure this works.
        d.create_dataset('a', data=np.arange(5))

if __name__ == '__main__':
    unittest.main()
