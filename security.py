from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
# 🔥 移除 passlib 的所有引用
import hashlib

# ============================================================
# 安全模块 (security.py)
# 负责 JWT 令牌的生成和验证，以及密码的哈希与校验。
# 不使用 passlib，改用内置 hashlib 实现简单 SHA256 哈希。
# 注意：此实现仅用于演示目的，生产环境应使用更安全的 bcrypt 并加盐。
# ============================================================

# JWT 配置
SECRET_KEY = "your-secret-key-keep-it-safe"   # 密钥，生产环境应从环境变量读取，不可硬编码
ALGORITHM = "HS256"                            # 签名算法，HS256 表示 HMAC-SHA256
ACCESS_TOKEN_EXPIRE_MINUTES = 3000              # 令牌有效期（分钟），此处为约2天（3000/24/60≈2.08）


def get_password_hash(password: str) -> str:
    """
    对明文密码进行哈希。
    使用 SHA256 算法，返回十六进制字符串。
    注意：此实现未加盐，因此相同的密码会产生相同的哈希值，存在彩虹表攻击风险。
          生产环境应使用 bcrypt 或 argon2 等加盐哈希算法。
    """
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证明文密码与哈希值是否匹配。
    将明文密码哈希后与存储的哈希值比较。
    """
    return get_password_hash(plain_password) == hashed_password


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    创建 JWT 访问令牌。

    :param data: 要编码到令牌中的数据，通常包含用户标识，如 {"sub": username}
    :param expires_delta: 可选的过期时间增量，若不提供则使用默认15分钟
    :return: 编码后的 JWT 字符串

    令牌中会包含标准字段 "exp"（过期时间，Unix 时间戳），由 python-jose 库自动处理格式。
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)  # 默认15分钟

    # 添加过期时间字段
    to_encode.update({"exp": expire})
    # 使用 python-jose 库编码 JWT
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt