```python
import hashlib
import random
import secrets
import string
from datetime import datetime

class UserAccount:
    def __init__(self, username: str):
        self.username = username
        self.password_hash = ""

    def set_password(self, password: str) -> None:
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()

    def check_password(self, password: str) -> bool:
        return self.password_hash == hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def get_current_date() -> str:
        return datetime.strftime(datetime.now(), "%Y-%m-%d")

    @staticmethod
    def generate_random_password() -> str:
        alphabet = string.ascii_letters + string.digits
        password = "".join(secrets.choice(alphabet) for _ in range(8))
        return password
```