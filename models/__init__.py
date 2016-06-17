# import pkgutil
# import sys
#
# def load_all_modules_from_dir(dirname):
#     for importer, package_name, _ in pkgutil.iter_modules([dirname]):
#         full_package_name = '%s.%s' % (dirname, package_name)
#         if full_package_name not in sys.modules:
#             module = importer.find_module(package_name).load_module(
#                 full_package_name)
#             print (module)
#
#
# print("module!!!!!!!")
# load_all_modules_from_dir('Foo')


# import os
# from models import *
# print("models:")
# __all__ = []
# for m in os.listdir("models/"):
#     if os.path.isfile(os.path.join("models/", m)) and m != "__init__.py":
#         print("add model: " + m)
#         __all__.append(os.path.splitext(m)[0])
# print(__all__)
# from models import cnn
# import models.cnn
# __all__ = ["cnn"]

# dynamically import all NN modules under models folder
import importlib
import os
from models import *
print("models:")
__all__ = []
for m in os.listdir("models/"):
    if os.path.isfile(os.path.join("models/", m)) and m != "__init__.py":
        print("add model: " + m)
        __all__.append(os.path.splitext(m)[0])
for m in __all__:
    importlib.import_module('models.' + m)
