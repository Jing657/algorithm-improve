# path
import pathlib

path = pathlib.Path(__file__).parent.parent.parent.resolve()
print(path)