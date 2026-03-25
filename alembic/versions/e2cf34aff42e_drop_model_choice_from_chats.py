"""drop model_choice from chats

Revision ID: e2cf34aff42e
Revises: e1a2b3c4d5e6
Create Date: 2026-03-25 11:01:11.721707

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e2cf34aff42e'
down_revision: Union[str, None] = 'e1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column('chats', 'model_choice')


def downgrade() -> None:
    op.add_column('chats', sa.Column(
        'model_choice',
        sa.VARCHAR(length=100),
        server_default=sa.text("'google/gemini-2.5-pro'"),
        nullable=False,
    ))
