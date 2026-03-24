from pydantic import BaseModel, EmailStr, Field, ConfigDict
from datetime import datetime
from typing import Optional


class UserRegister(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=6)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UpdateApiKeyRequest(BaseModel):
    openrouter_api_key: str = Field(min_length=10)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    name: str
    created_at: datetime
    has_openrouter_key: bool = False


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: int
    title: str
    app_url: Optional[str] = None
    deployed_url: Optional[str] = None
    model_choice: str
    created_at: datetime


class ProjectsListResponse(BaseModel):
    projects: list[ProjectResponse]


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class RegisterResponse(BaseModel):
    """Response model for successful registration"""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse