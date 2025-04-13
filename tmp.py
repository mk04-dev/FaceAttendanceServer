import requests

# Tạo session
session = requests.Session()

cookie = ''
# Headers
headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "vi-VN,vi;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "Content-Type": "application/x-www-form-urlencoded",
}

# Dữ liệu đăng nhập (form-urlencoded)
payload = {
    "USERNAME": "admin",
    "PASSWORD": "111111",
    'userTenantId': 'test1',
    'JavaScriptEnabled': 'Y',
}

# # # Gửi POST request
response = session.post(
    "https://tenant1.ecom365.localhost:8443/catalog/control/login",
    headers=headers,
    data=payload,
    verify=False  # Nếu là localhost và dùng HTTPS tự ký, thêm dòng này để tránh lỗi SSL
)

for key, value in session.cookies.get_dict().items():
    cookie += f"{key}={value}; "

print(cookie)
shiftResult = session.get(
    "https://localhost:8443/erp/fdn/v1/api/userLogin/permissions",
    headers={
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "vi-VN,vi;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5",
        "Cookie": cookie,
    },
    verify=False  # Nếu là localhost và dùng HTTPS tự ký, thêm dòng này để tránh lỗi SSL
)


print(shiftResult.status_code)
print(shiftResult.text)