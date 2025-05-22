from io import BytesIO
from session.session_manager import SessionManager
from utils import get_tenant_info
import json
import requests

def handle_response(response):
    if response.status_code != 200:
        raise Exception(response.text)
    
    res = response.json()
    if res.get("statusCode") != 200:
        raise Exception(res.get('message'))
    
    return res

async def load_all_embeddings(tenant_cd: str):
    """
    Load all embeddings from the database.
    """
    HOST = get_tenant_info(tenant_cd)  # Kiểm tra xem tenant_cd có hợp lệ không
    session = SessionManager.get_session(tenant_cd)

    url = f"{HOST}/erp/hrm/v1/api/personEmbedding"
    
    response = session.get(url)
    res = handle_response(response)
    return res.get("data", [])

async def create_timekeeping(tenant_cd: str, party_id, address, branch_id, image_bytes):
    HOST = get_tenant_info(tenant_cd)  # Kiểm tra xem tenant_cd có hợp lệ không
    session = SessionManager.get_session(tenant_cd)
    url = f"{HOST}/erp/hrm/v1/api/timekeeping/create"
    attendance_dto = {
        "partyId": party_id,
        "address": address,
        "branchId": branch_id,
        "note": "Face recognition",
        "sourceTimekeeping": "Face recognition",
    }
    dto_part = ('attendanceDTO', (None, json.dumps(attendance_dto), 'application/json'))
    image_file = BytesIO(image_bytes)
    image_file.name = "face.jpg"  # cần gán tên file
    files = [
        dto_part,
        ('image', (image_file.name, image_file, 'image/jpeg'))
    ]
    response = session.post(url, files=files)
    res = handle_response(response)
    return res

async def create_person_embedding(tenant_cd: str, data, token):
    HOST = get_tenant_info(tenant_cd)  # Kiểm tra xem tenant_cd có hợp lệ không
    url = f"{HOST}/erp/hrm/v1/api/personEmbedding"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    response = requests.post(url, json=data, headers=headers)
    res = handle_response(response)
    return res.get("data", [])

async def delete_person_embedding(tenant_cd: str, party_id, token):
    HOST = get_tenant_info(tenant_cd)  # Kiểm tra xem tenant_cd có hợp lệ không
    url = f"{HOST}/erp/hrm/v1/api/personEmbedding/{party_id}"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    response = requests.delete(url, headers=headers)
    handle_response(response)

async def checkInByFaceRecognition(tenant_cd: str, party_id, address, branch_id, image_bytes, token):
    HOST = get_tenant_info(tenant_cd)  # Kiểm tra xem tenant_cd có hợp lệ không
    attendanceDTO = {
        "partyIds": [party_id],
        "address": address,
        "note": "Face recognition",
        "sourceTimekeeping": "Face recognition",
        "branchId": branch_id,
    }
    files = {
        'attendanceDTO': (None, json.dumps(attendanceDTO), 'application/json'),
        'image': ('image.jpg', image_bytes, 'image/jpeg')
    }
    headers={
        "Authorization": f"Bearer {token}"
    }
    url = f"{HOST}/erp/hrm/v1/api/attendance/checkInByFaceRecognition"
    response = requests.post(url, files=files, headers=headers)
    res = handle_response(response)
    return res.get("data", [])

    