from fastapi import Header


async def get_subject(x_user_id: str | None = Header(default=None)) -> str:
    user_id = x_user_id or "dj"
    return f"user:{user_id}"
