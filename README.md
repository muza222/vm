# vm
python virtual machine implementation
In this project, we were trying to write a virtual machine that executes python byte code on python. 

As an example
DIS documentation: https://docs.python.org/release/3.10.6/library/dis.html#module-dis . It describes all existing python byte code operations.
An academic interpreter project for PY27 and PY33, provided with a lot of comments, but not without problems: https://github.com/nedbat/byterun .
His detailed discussion in the blog: http://www.aosabook.org/en/500L/a-python-interpreter-written-in-python.html .
The source code of the CPython interpreter will help you figure out the subtleties: https://github.com/python/cpython/blob/3.10/Python/ceval.c .
