from consts import TENANT_DICT

def get_tenant_info(tenant_id):
    """Lấy thông tin tenant từ TENANT_DICT"""
    if tenant_id not in TENANT_DICT:
        raise ValueError(f"Tenant ID [{tenant_id}] not found")
    return TENANT_DICT[tenant_id]


def check_permission(user_permissions, required_permissions):
    for user_permission in user_permissions:
        if user_permission["groupId"] == 'FULLADMIN' \
            or user_permission["groupId"] in required_permissions:
            return True
    return False