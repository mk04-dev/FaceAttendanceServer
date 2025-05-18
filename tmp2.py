

import requests
url = 'https://demo.ecom365.vn/erp/fdn/api/userLogin/basic'
res = requests.get(url, headers={
    "Content-Type": "application/json",
    "Bearer": "eyJhbGciOiJIUzUxMiJ9.eyJpYXQiOjE3NDc1NzMxODQsImV4cCI6MTc0NzU4NzU5OSwidXNlckxvZ2luSWQiOiJhZG1pbiIsImRlbGVnYXRvck5hbWUiOiJkZWZhdWx0I3RlbmFudDQwIiwiZGVsZWdhdG9yVGVuYW50SWQiOiJ0ZW5hbnQ0MCIsInNlcnZlck5hbWUiOiJkZW1vLmVjb20zNjUudm4iLCJkZWxlZ2F0b3JUZW5hbnRDb21wYW55Q29kZSI6ImRlbW8iLCJwYXJ0eUlkIjoiYWRtaW4ifQ.foP1bKkmRqDGy6zMe92xhoYkGFaEzENqhFuJ98yJ-6LDwmuPX0TK_avS6dAkjbWzGkE3-DqKnbL-nFym5Wtlbw",
})
print(res.text)