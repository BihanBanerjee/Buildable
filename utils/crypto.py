import base64
import hashlib
from cryptography.fernet import Fernet
from auth.utils import SECRET_KEY

# Derive a Fernet-compatible 32-byte key from SECRET_KEY
_fernet_key = base64.urlsafe_b64encode(hashlib.sha256(SECRET_KEY.encode()).digest())
_fernet = Fernet(_fernet_key)


def encrypt_api_key(plaintext: str) -> str:
    return _fernet.encrypt(plaintext.encode()).decode()


def decrypt_api_key(ciphertext: str) -> str:
    return _fernet.decrypt(ciphertext.encode()).decode()
