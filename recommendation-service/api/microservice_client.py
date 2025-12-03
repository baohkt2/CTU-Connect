import httpx
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from config.settings import config

logger = logging.getLogger(__name__)

class MicroserviceClient:
    """Client để giao tiếp với các microservices khác trong hệ thống CTU Connect"""

    def __init__(self):
        self.timeout = httpx.Timeout(config.REQUEST_TIMEOUT)
        self.session = None

    async def initialize(self):
        """Khởi tạo HTTP client session"""
        self.session = httpx.AsyncClient(timeout=self.timeout)

    async def close(self):
        """Đóng HTTP client session"""
        if self.session:
            await self.session.aclose()

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin profile người dùng từ User Service"""
        try:
            url = f"{config.USER_SERVICE_URL}/api/users/{user_id}/profile"
            response = await self.session.get(url)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.warning(f"User profile not found: {user_id}")
                return None
            else:
                logger.error(f"Error fetching user profile: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error connecting to user service: {e}")
            return None

    async def get_user_interactions(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Lấy lịch sử tương tác của người dùng"""
        try:
            url = f"{config.USER_SERVICE_URL}/api/users/{user_id}/interactions"
            params = {"limit": limit}
            response = await self.session.get(url, params=params)

            if response.status_code == 200:
                return response.json().get("interactions", [])
            else:
                logger.error(f"Error fetching user interactions: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error fetching user interactions: {e}")
            return []

    async def get_posts_batch(self, post_ids: List[str]) -> List[Dict[str, Any]]:
        """Lấy thông tin nhiều bài viết từ Post Service"""
        try:
            url = f"{config.POST_SERVICE_URL}/api/posts/batch"
            data = {"postIds": post_ids}
            response = await self.session.post(url, json=data)

            if response.status_code == 200:
                return response.json().get("posts", [])
            else:
                logger.error(f"Error fetching posts batch: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error fetching posts batch: {e}")
            return []

    async def get_trending_posts(self, category: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Lấy danh sách bài viết trending từ Post Service"""
        try:
            url = f"{config.POST_SERVICE_URL}/api/posts/trending"
            params = {"limit": limit}
            if category:
                params["category"] = category

            response = await self.session.get(url, params=params)

            if response.status_code == 200:
                return response.json().get("posts", [])
            else:
                logger.error(f"Error fetching trending posts: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error fetching trending posts: {e}")
            return []

    async def get_posts_by_category(self, category: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Lấy bài viết theo category"""
        try:
            url = f"{config.POST_SERVICE_URL}/api/posts/category/{category}"
            params = {"limit": limit}
            response = await self.session.get(url, params=params)

            if response.status_code == 200:
                return response.json().get("posts", [])
            else:
                logger.error(f"Error fetching posts by category: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error fetching posts by category: {e}")
            return []

    async def validate_user_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Xác thực token người dùng với Auth Service"""
        try:
            url = f"{config.AUTH_SERVICE_URL}/api/auth/validate"
            headers = {"Authorization": f"Bearer {token}"}
            response = await self.session.post(url, headers=headers)

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Token validation failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return None

    async def send_interaction_event(self, user_id: str, post_id: str,
                                   interaction_type: str, context: Dict[str, Any]):
        """Gửi sự kiện tương tác đến các services khác"""
        try:
            # Gửi đến User Service để cập nhật profile
            user_url = f"{config.USER_SERVICE_URL}/api/users/{user_id}/interactions"
            user_data = {
                "postId": post_id,
                "interactionType": interaction_type,
                "timestamp": datetime.utcnow().isoformat(),
                "context": context
            }

            # Gửi đến Post Service để cập nhật metrics
            post_url = f"{config.POST_SERVICE_URL}/api/posts/{post_id}/interactions"
            post_data = {
                "userId": user_id,
                "interactionType": interaction_type,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Gửi song song
            responses = await asyncio.gather(
                self.session.post(user_url, json=user_data),
                self.session.post(post_url, json=post_data),
                return_exceptions=True
            )

            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Error sending interaction event {i}: {response}")
                elif response.status_code != 200:
                    logger.warning(f"Interaction event {i} failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Error sending interaction events: {e}")

# Global client instance
microservice_client = MicroserviceClient()
