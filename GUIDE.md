1. Use Imperative for comments, [e.g. from python mailing list](https://mail.python.org/pipermail/python-list/2014-February/666361.html)
  - Use [numpy style documentation](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)
2. make sure you don't have lint errors, run `flake8` to see errors, most of them are possible to fix with autopep8
```
autopep8 --in-place --aggressive --aggressive FILENAME.py
```
