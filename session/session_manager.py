import requests
import jwt
from threading import Lock
from datetime import datetime, timedelta
from session.session_wrapper import SessionWrapper
from utils import get_tenant_info

class SessionManager:
    _sessions = {}
    _lock = Lock()

    @staticmethod
    def _decode_exp_from_token(token: str) -> datetime:
        try:
            # Decode payload mà không cần verify signature (vì chỉ cần 'exp')
            decoded = jwt.decode(token, options={"verify_signature": False})
            exp_timestamp = decoded.get("exp")

            if not exp_timestamp:
                raise ValueError("exp not found in token")

            # Trừ đi buffer 60 giây
            return datetime.utcfromtimestamp(exp_timestamp) - timedelta(seconds=60)

        except Exception as e:
            raise Exception(f"Failed to decode token: {e}")

    @staticmethod
    def _create_session(tenant_cd: str) -> SessionWrapper:
        TENANT = get_tenant_info(tenant_cd)

        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "vi-VN,vi;q=0.9,fr-FR;q=0.8,fr;q=0.7,en-US;q=0.6,en;q=0.5",
            "Cache-Control": "max-age=0",
            "Connection": "keep-alive",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        
        data = {
            "USERNAME": TENANT["username"],
            "PASSWORD": TENANT["password"],
        }
        url = f"{TENANT['host']}/webpos/control/ecomGetAuthenticationToken"
        
        auth_response = requests.post(url, headers=headers, data=data, timeout=10, allow_redirects=False)
        if auth_response.status_code != 200:
            raise Exception(f"Auth failed for tenant {tenant_cd}: {auth_response.text}")

        token = auth_response.json().get("token")

        session = requests.Session()
        session.headers.update({"Bearer": token})

        token_expiry = SessionManager._decode_exp_from_token(token)
        return SessionWrapper(session, token_expiry)

    @classmethod
    def get_session(cls, tenant_cd: str) -> requests.Session:
        with cls._lock:
            wrapper = cls._sessions.get(tenant_cd)

            if wrapper is None or wrapper.is_expired():
                wrapper = cls._create_session(tenant_cd)
                cls._sessions[tenant_cd] = wrapper

            return wrapper.session
