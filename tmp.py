import requests
import json
# Tạo session
session = requests.Session()

# Headers
headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "vi-VN,vi;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
}

res = session.post('https://localhost:8443/webpos/control/ecomGetAuthenticationToken', 
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Accept-Language": "vi-VN,vi;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5",
                "Cache-Control": "max-age=0",
                "Connection": "keep-alive",
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Tenant-Id": "test1",
            }, 
            data={
                "USERNAME": "admin",
                "PASSWORD": "111111",
            },
            verify=False)
print(res.json().get('token'))

# # Dữ liệu đăng nhập (form-urlencoded)
# payload = {"search":"","page":1,"limit":10}

# # # # Gửi POST request
# response = session.get(
#     "https://localhost:8443/erp/hrm/v1/api/setting/timekeeping",
#     headers=headers,
#     verify=False  # Nếu là localhost và dùng HTTPS tự ký, thêm dòng này để tránh lỗi SSL
# )
# print(response.text)