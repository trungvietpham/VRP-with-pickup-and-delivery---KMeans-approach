# 1. Cài môi trường:
## Cài môi trường conda
```sh
conda create -n aproject python=3.10.4 
```
```sh
conda activate aproject
```

## Cài các thư viện cần thiết: 
```sh
pip install -r src/requirements.txt
```

# 2. Cấu hình file config
Đường dẫn tới file config: [Here](src/config/config.yaml)     
Các thông tin chi tiết về file config: [Xem ở đây](https://docs.google.com/document/d/1AH7FLp9vKLVGLRO_chTXs87RuDpbIY1-/edit?usp=share_link&ouid=100840157907827359792&rtpof=true&sd=true)

# 3. Chạy thuật toán:
Sau khi cấu hình xong file config, ta chạy thuật toán:    
Mở file [main](src/main.py) và chạy file    
Sau khi kết thúc quá trình chạy, ta sẽ thấy thông tin chạy trong thư mục [scenarios](scenarios), thông tin cụ thể về folder cũng sẽ được in ra ở terminal