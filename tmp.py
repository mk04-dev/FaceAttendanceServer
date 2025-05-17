import requests
# Tạo session
session = requests.Session()
res = session.post('https://demo.ecom365.vn/webpos/control/ecomGetAuthenticationToken', 
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Accept-Language": "vi-VN,vi;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5",
                "Cache-Control": "max-age=0",
                "Connection": "keep-alive",
                "Content-Type": "application/x-www-form-urlencoded",
                # "X-Tenant-Id": "test1",
            }, 
            data={
                "USERNAME": "admin",
                "PASSWORD": "111111",
            })
print(res.json())

headers = {
    "Bearer": res.json().get('token'),
    "Content-Type": "application/json;charset=UTF-8",
}
# # # Gửi POST request
response = session.get("https://demo.ecom365.vn/erp/hrm/v1/api/personEmbedding", headers=headers)
print(response.text)