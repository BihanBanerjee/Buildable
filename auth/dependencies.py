from fastapi import Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from db.models import User
from db.base import get_db
from .utils import decode_token



security = HTTPBearer()
# This tells FastAPI: "expect an Authorization: Bearer <token> header on this request." It's a FastAPI security scheme that automatically extracts the token from the header.


async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        # Depends(security) means FastAPI runs security first and injects its result into credentials.

        # HTTPAuthorizationCredentials is an object with exactly 2 fields:
        # credentials.scheme       # → "Bearer"
        # credentials.credentials  # → "eyJhbGciOiJIUzI1NiIs..."  (the actual JWT token)

        db: AsyncSession = Depends(get_db),
) -> User:
    """Get the current authenticated user from the token"""

    token = credentials.credentials
    payload = decode_token(token=token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="could not validate credentials",
        )

    user_id_str = payload.get("sub")
    if not user_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="could not validate credentials",
        )
    
    try:
        user_id = int(user_id_str)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID format",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
        )

    return user