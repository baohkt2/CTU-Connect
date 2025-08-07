import asyncio
import httpx
import logging
from typing import Dict, Optional, List
import json
from datetime import datetime

from config.settings import config

logger = logging.getLogger(__name__)

class EurekaClient:
    """Client để đăng ký và discovery services với Eureka Server"""
    
    def __init__(self):
        self.eureka_url = config.EUREKA_SERVER_URL.rstrip('/eureka')
        self.service_name = config.SERVICE_NAME
        self.service_id = config.SERVICE_ID
        self.host = config.HOST
        self.port = config.PORT
        self.session = None
        self.registered = False
        
    async def initialize(self):
        """Khởi tạo Eureka client"""
        self.session = httpx.AsyncClient(timeout=30.0)
        
    async def close(self):
        """Đóng Eureka client"""
        if self.registered:
            await self.deregister()
        if self.session:
            await self.session.aclose()
    
    async def register(self):
        """Đăng ký service với Eureka"""
        try:
            registration_data = {
                "instance": {
                    "instanceId": self.service_id,
                    "hostName": self.host,
                    "app": self.service_name.upper(),
                    "ipAddr": self.host,
                    "port": {
                        "$": self.port,
                        "@enabled": "true"
                    },
                    "securePort": {
                        "$": 443,
                        "@enabled": "false"
                    },
                    "status": "UP",
                    "overriddenstatus": "UNKNOWN",
                    "healthCheckUrl": f"http://{self.host}:{self.port}/health",
                    "statusPageUrl": f"http://{self.host}:{self.port}/health",
                    "homePageUrl": f"http://{self.host}:{self.port}/",
                    "vipAddress": self.service_name,
                    "secureVipAddress": self.service_name,
                    "isCoordinatingDiscoveryServer": "false",
                    "lastUpdatedTimestamp": str(int(datetime.now().timestamp() * 1000)),
                    "lastDirtyTimestamp": str(int(datetime.now().timestamp() * 1000)),
                    "actionType": "ADDED",
                    "metadata": {
                        "@class": "java.util.Collections$EmptyMap",
                        "management.port": str(self.port)
                    },
                    "dataCenterInfo": {
                        "@class": "com.netflix.appinfo.InstanceInfo$DefaultDataCenterInfo",
                        "name": "MyOwn"
                    }
                }
            }
            
            url = f"{self.eureka_url}/eureka/apps/{self.service_name.upper()}"
            headers = {"Content-Type": "application/json"}
            
            response = await self.session.post(url, json=registration_data, headers=headers)
            
            if response.status_code in [200, 204]:
                self.registered = True
                logger.info(f"Successfully registered {self.service_name} with Eureka")
                
                # Bắt đầu heartbeat
                asyncio.create_task(self._heartbeat_loop())
                return True
            else:
                logger.error(f"Failed to register with Eureka: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering with Eureka: {e}")
            return False
    
    async def deregister(self):
        """Hủy đăng ký service với Eureka"""
        try:
            url = f"{self.eureka_url}/eureka/apps/{self.service_name.upper()}/{self.service_id}"
            response = await self.session.delete(url)
            
            if response.status_code in [200, 204]:
                logger.info(f"Successfully deregistered {self.service_name} from Eureka")
                self.registered = False
            else:
                logger.warning(f"Failed to deregister from Eureka: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error deregistering from Eureka: {e}")
    
    async def send_heartbeat(self):
        """Gửi heartbeat đến Eureka"""
        try:
            url = f"{self.eureka_url}/eureka/apps/{self.service_name.upper()}/{self.service_id}"
            response = await self.session.put(url)
            
            if response.status_code == 200:
                logger.debug("Heartbeat sent successfully")
                return True
            elif response.status_code == 404:
                logger.warning("Service not found in Eureka, re-registering...")
                return await self.register()
            else:
                logger.warning(f"Heartbeat failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
            return False
    
    async def _heartbeat_loop(self):
        """Vòng lặp gửi heartbeat định kỳ"""
        while self.registered:
            try:
                await asyncio.sleep(30)  # Heartbeat mỗi 30 giây
                if self.registered:
                    await self.send_heartbeat()
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                break
    
    async def discover_service(self, service_name: str) -> Optional[Dict]:
        """Tìm kiếm service thông qua Eureka"""
        try:
            url = f"{self.eureka_url}/eureka/apps/{service_name.upper()}"
            headers = {"Accept": "application/json"}
            
            response = await self.session.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                instances = data.get("application", {}).get("instance", [])
                
                # Nếu chỉ có 1 instance, Eureka trả về dict thay vì list
                if isinstance(instances, dict):
                    instances = [instances]
                
                # Lọc chỉ các instance đang UP
                healthy_instances = [
                    instance for instance in instances
                    if instance.get("status") == "UP"
                ]
                
                return healthy_instances
            else:
                logger.warning(f"Service {service_name} not found in Eureka")
                return None
                
        except Exception as e:
            logger.error(f"Error discovering service {service_name}: {e}")
            return None
    
    async def get_service_url(self, service_name: str) -> Optional[str]:
        """Lấy URL của một service"""
        instances = await self.discover_service(service_name)
        
        if instances:
            # Chọn instance đầu tiên (có thể implement load balancing sau)
            instance = instances[0]
            host = instance.get("ipAddr") or instance.get("hostName")
            port = instance.get("port", {}).get("$", 8080)
            return f"http://{host}:{port}"
        
        return None
    
    async def get_all_services(self) -> List[Dict]:
        """Lấy danh sách tất cả services đã đăng ký"""
        try:
            url = f"{self.eureka_url}/eureka/apps"
            headers = {"Accept": "application/json"}
            
            response = await self.session.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                applications = data.get("applications", {}).get("application", [])
                
                if isinstance(applications, dict):
                    applications = [applications]
                
                return applications
            else:
                logger.error(f"Failed to get services from Eureka: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting services from Eureka: {e}")
            return []

# Global Eureka client instance
eureka_client = EurekaClient()
