from .base import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from sqlalchemy import Integer, String, DateTime, ForeignKey, Text, JSON


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Rate limiting fields
    last_query_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    # Token/Credits System - user gets 2 tokens per day
    tokens_remaining: Mapped[int] = mapped_column(Integer, default=2)
    tokens_reset_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    # A user can have many Chats.
    # back_populates="user" links back to the user field in the Chat model.
    # cascade="all, delete-orphan" → if a user is deleted, all their chats are deleted too (prevents orphaned chats).
    chats: Mapped[List["Chat"]] = relationship(
        "Chat", back_populates="user", cascade="all, delete-orphan"
    )

    
    def can_make_query(self) -> bool:
        """Check if user can make a query based on rate limiting and tokens"""
        # Special user gets unlimited access
        if self.email == "banerjeebihan456@gmail.com":
            return True
        
        # Check if we need to reset tokens (24 hours passed)
        if self.tokens_reset_at is None or datetime.now(timezone.utc) >= self.tokens_reset_at:
            # Reset tokens
            self.tokens_remaining = 2
            self.tokens_reset_at = datetime.now(timezone.utc) + timedelta(hours=24)
            return True
        
        return self.tokens_remaining > 0
    

    def use_token(self) -> bool:
        """Use one token and return True if successful, False if no tokens left"""
        if self.email == "banerjeebihan456@gmail.com":
            return True
        
        if self.tokens_reset_at is None or datetime.now(timezone.utc) >= self.tokens_reset_at:
            self.tokens_remaining = 2
            self.tokens_reset_at = datetime.now(timezone.utc) + timedelta(hours=24)
        
        if self.tokens_remaining > 0:
            self.tokens_remaining -= 1
            self.last_query_at = datetime.now(timezone.utc)
            return True
        return False
    
    
    def get_time_until_reset(self) -> float:
        """Get hours until token reset"""
        if self.tokens_reset_at is None:
            return 0
        time_diff = self.tokens_reset_at - datetime.now(timezone.utc)
        return max(0, time_diff.total_seconds() / 3600)
    
    def update_last_query(self):
        """Update the last query timestamp"""
        self.last_query_at = datetime.now(timezone.utc)

class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[int] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE")
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    app_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    user: Mapped["User"] = relationship("User", back_populates="chats")
    messages: Mapped[List["Message"]] = relationship(
        "Message", back_populates="chat", cascade="all, delete-orphan", order_by="Message.created_at"
    )



class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(String(36), primary_key=True)
    chat_id: Mapped[int] = mapped_column(
        String(36), ForeignKey("chats.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(50)) # 'user' or 'assistant'
    content: Mapped[str] = mapped_column(Text) # Use Text for unlimited size
    event_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True) # 'message_start', 'message_end', or None for regular messages. For system events like 'builder_started'
    tool_calls: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True) # Store tool calls as JSON: [{name: str, status: 'success'|'error', output: str}]

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    chat: Mapped["Chat"] = relationship("Chat", back_populates="messages")
