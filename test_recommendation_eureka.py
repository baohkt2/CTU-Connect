#!/usr/bin/env python3
"""
Script để test kết nối Eureka của Recommendation Service
"""

import requests
import time
import json
from typing import Dict, Any

def check_eureka_server(eureka_url: str = "http://localhost:8761") -> bool:
    """Kiểm tra Eureka server có hoạt động không"""
    try:
        response = requests.get(f"{eureka_url}/actuator/health", timeout=5)
        if response.status_code == 200:
            print("✅ Eureka server is running")
            return True
        else:
            print(f"❌ Eureka server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Eureka server: {e}")
        return False

def check_recommendation_service(service_url: str = "http://localhost:8086") -> bool:
    """Kiểm tra Recommendation service có hoạt động không"""
    try:
        response = requests.get(f"{service_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Recommendation service is running")
            print(f"   Service: {health_data.get('service', 'Unknown')}")
            print(f"   Status: {health_data.get('status', 'Unknown')}")
            print(f"   Eureka Enabled: {health_data.get('eureka_enabled', 'Unknown')}")
            print(f"   Eureka Registered: {health_data.get('eureka_registered', 'Unknown')}")

            last_error = health_data.get('last_eureka_error')
            if last_error:
                print(f"   Last Eureka Error: {last_error}")

            return True
        else:
            print(f"❌ Recommendation service health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Recommendation service: {e}")
        return False

def check_eureka_apps(eureka_url: str = "http://localhost:8761") -> Dict[str, Any]:
    """Kiểm tra các ứng dụng đã đăng ký với Eureka"""
    try:
        response = requests.get(f"{eureka_url}/eureka/apps",
                              headers={"Accept": "application/json"},
                              timeout=5)
        if response.status_code == 200:
            apps_data = response.json()
            applications = apps_data.get('applications', {}).get('application', [])

            print(f"\n📋 Registered applications in Eureka:")
            if not applications:
                print("   No applications registered")
                return {}

            registered_services = {}
            for app in applications:
                app_name = app.get('name', 'Unknown')
                instances = app.get('instance', [])
                if isinstance(instances, dict):
                    instances = [instances]

                print(f"\n   🔹 {app_name}")
                registered_services[app_name] = []

                for instance in instances:
                    status = instance.get('status', 'Unknown')
                    host = instance.get('hostName', 'Unknown')
                    port = instance.get('port', {}).get('$', 'Unknown')
                    home_url = instance.get('homePageUrl', 'Unknown')

                    print(f"      - Status: {status}")
                    print(f"      - Host: {host}:{port}")
                    print(f"      - URL: {home_url}")

                    registered_services[app_name].append({
                        'status': status,
                        'host': host,
                        'port': port,
                        'url': home_url
                    })

            return registered_services
        else:
            print(f"❌ Failed to get Eureka applications: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Error getting Eureka applications: {e}")
        return {}

def wait_for_registration(max_wait: int = 120) -> bool:
    """Đợi recommendation service đăng ký với Eureka"""
    print(f"\n⏳ Waiting for recommendation service to register with Eureka (max {max_wait}s)...")

    start_time = time.time()
    while time.time() - start_time < max_wait:
        apps = check_eureka_apps()
        if 'RECOMMENDATION-SERVICE' in apps:
            instances = apps['RECOMMENDATION-SERVICE']
            up_instances = [i for i in instances if i['status'] == 'UP']
            if up_instances:
                print("✅ Recommendation service successfully registered with Eureka!")
                return True

        print("   Still waiting...")
        time.sleep(10)

    print("❌ Timeout waiting for recommendation service registration")
    return False

def main():
    print("🔍 Testing Recommendation Service Eureka Registration")
    print("=" * 60)

    # 1. Check Eureka server
    print("\n1. Checking Eureka Server...")
    if not check_eureka_server():
        print("❌ Please start Eureka server first")
        return

    # 2. Check Recommendation service
    print("\n2. Checking Recommendation Service...")
    if not check_recommendation_service():
        print("❌ Please start Recommendation service first")
        return

    # 3. Check registered applications
    print("\n3. Checking Eureka Applications...")
    apps = check_eureka_apps()

    # 4. Check if recommendation service is registered
    if 'RECOMMENDATION-SERVICE' in apps:
        print("\n✅ SUCCESS: Recommendation service is registered with Eureka!")
        instances = apps['RECOMMENDATION-SERVICE']
        for instance in instances:
            if instance['status'] == 'UP':
                print(f"   Active instance: {instance['url']}")
    else:
        print("\n⏳ Recommendation service not yet registered, waiting...")
        if wait_for_registration():
            print("🎉 Registration completed successfully!")
        else:
            print("❌ Registration failed or timed out")

    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    main()
