# coalescing_graph_with_PTX
## Requirements
pip install -r requirements.txt
## Usage
### Step 0
make PTX code from execution file
```bash
cuobjdump -ptx EXECTION_FILE
```
And name the ptx file with thread block id and thread id
ex) GEMM_2_8_32_8.ptx
### Step 1
make syntax tree based on PTX code
```bash
python3 locality_guru.py -f FILE_NAME
```
### Step 2
```bash
python3 locality_map_coalescing.py -f FILE_NAME
```
The result is saved in result_img directory

