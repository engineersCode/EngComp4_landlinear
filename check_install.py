import importlib
from sys import version_info
from distutils.version import StrictVersion

def compare_version(module, version):
    version_loaded = importlib.import_module(module).__version__
    if StrictVersion(version_loaded) < StrictVersion(version):
        return 'current version {}, please use conda or pip to upgrade to {} or later'.format(version_loaded, version)
    return None

def main():
    try:
        assert version_info >= (3,5)
    except AssertionError:
        print('This tutorial requires Python version 3.5 or later.')

    modules = {'numpy': '1.15.1',
               'matplotlib': '2.0.2',
               'jupyter': None,
               'sympy': None,
               'imageio': None,
               'ipywidgets': None}

    for module, version in modules.items():
        spec = importlib.util.find_spec(module)
        if spec is None:
            print('{} ... not found, use conda or pip to install'.format(module))
        else:
            if version is not None:
                msg = compare_version(module, version)
                if msg is not None:
                    print('{} ... {}'.format(module, msg))
                else:
                    print('{} ... OK'.format(module))
            else:
                print('{} ... OK'.format(module))

if __name__ == '__main__':
    main()
