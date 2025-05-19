from io import BytesIO
from session_manager import SessionManager
from utils import get_tenant_info
import json
from logger_service import LoggerService
LOGGER = LoggerService().get_logger()
import requests
def handle_response(response):
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.text}")
    
    res = response.json()
    if res.get("statusCode") != 200:
        raise Exception(f"Request failed: {res.get('message')}")
    
    return res

async def load_all_embeddings(tenant_cd: str):
    """
    Load all embeddings from the database.
    """
    TENANT = get_tenant_info(tenant_cd)  # Kiểm tra xem tenant_cd có hợp lệ không
    session = SessionManager.get_session(tenant_cd)

    url = f"{TENANT['host']}/erp/hrm/v1/api/personEmbedding"
    
    response = session.get(url)
    res = handle_response(response)
    return res.get("data", [])

async def create_timekeeping(tenant_cd: str, party_id, address, branch_id, image_bytes):
    TENANT = get_tenant_info(tenant_cd)  # Kiểm tra xem tenant_cd có hợp lệ không
    session = SessionManager.get_session(tenant_cd)
    url = f"{TENANT['host']}/erp/hrm/v1/api/timekeeping/create"
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

async def create_person_embedding(tenant_cd: str, data):
    TENANT = get_tenant_info(tenant_cd)  # Kiểm tra xem tenant_cd có hợp lệ không
    session = SessionManager.get_session(tenant_cd)
    url = f"{TENANT['host']}/erp/hrm/v1/api/personEmbedding"
    headers = {"Content-Type": "application/json"}
    response = session.post(url, json=data, headers=headers)
    res = handle_response(response)
    return res.get("data", [])

async def delete_person_embedding(tenant_cd: str, party_id):
    TENANT = get_tenant_info(tenant_cd)  # Kiểm tra xem tenant_cd có hợp lệ không
    session = SessionManager.get_session(tenant_cd)
    url = f"{TENANT['host']}/erp/hrm/v1/api/personEmbedding/{party_id}"
    response = session.delete(url)
    handle_response(response)

async def checkInByFaceRecognition(tenant_cd: str, party_id, address, branch_id, image_bytes):
    TENANT = get_tenant_info(tenant_cd)  # Kiểm tra xem tenant_cd có hợp lệ không
    session = SessionManager.get_session(tenant_cd)


    attendanceDTO = {
        "partyIds": [party_id],
        "address": address,
        "note": "Face recognition",
        "sourceTimekeeping": "Face recognition",
        # "appInstallationId": "",
        # "registerBusinessTripId": "",
        # "contentId": "",
        # "executionTime": "",
        "branchId": branch_id,
    }
    files = {
        'attendanceDTO': (None, json.dumps(attendanceDTO), 'application/json'),
        'image': ('image.jpg', image_bytes, 'image/jpeg')
    }
    url = f"{TENANT['host']}/erp/hrm/v1/api/attendance/checkInByFaceRecognition"
    response = session.post(url, files=files)
    handle_response(response)