rm libeval.so
nvcc -Xcompiler -fPIC -shared -o libeval.so libeval.cu
venv/bin/python run.py "$@"