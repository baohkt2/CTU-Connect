"""
Test script for Phase 1 endpoints
Tests:
1. /embed/user - Single user embedding with caching
2. /similarity/users - Batch user similarity computation
"""

import requests
import json

BASE_URL = "http://localhost:8000"

# Test data
test_user_current = {
    "user_id": "user_001",
    "major": "Công nghệ thông tin",
    "faculty": "Công nghệ thông tin và Truyền thông",
    "courses": ["Lập trình Python", "Machine Learning", "Data Science"],
    "skills": ["Python", "TensorFlow", "Docker"],
    "bio": "Sinh viên năm 3 chuyên ngành AI"
}

test_user_candidate1 = {
    "user_id": "user_002",
    "major": "Công nghệ thông tin",
    "faculty": "Công nghệ thông tin và Truyền thông",
    "courses": ["Lập trình Java", "Machine Learning", "Big Data"],
    "skills": ["Java", "Spark", "Hadoop"],
    "bio": "Sinh viên năm 4 chuyên ngành Big Data"
}

test_user_candidate2 = {
    "user_id": "user_003",
    "major": "Kỹ thuật phần mềm",
    "faculty": "Công nghệ thông tin và Truyền thông",
    "courses": ["Web Development", "Mobile App", "DevOps"],
    "skills": ["React", "Node.js", "AWS"],
    "bio": "Sinh viên năm 3 chuyên ngành Web Development"
}

print("="*60)
print("PHASE 1 ENDPOINT TESTING")
print("="*60)

# Test 1: Single user embedding (first call - should generate)
print("\n1. Testing /embed/user (First call - GENERATE)")
print("-"*60)
response1 = requests.post(
    f"{BASE_URL}/embed/user",
    json=test_user_current
)
if response1.status_code == 200:
    result1 = response1.json()
    print(f"✅ Status: {response1.status_code}")
    print(f"   User ID: {result1['user_id']}")
    print(f"   Dimension: {result1['dimension']}")
    print(f"   Source: {result1['source']}")
    print(f"   Profile Hash: {result1['profile_hash'][:16]}...")
    print(f"   Embedding (first 5): {result1['embedding'][:5]}")
else:
    print(f"❌ Error: {response1.status_code}")
    print(f"   {response1.text}")

# Test 2: Same user embedding (second call - should use cache)
print("\n2. Testing /embed/user (Second call - CACHE)")
print("-"*60)
response2 = requests.post(
    f"{BASE_URL}/embed/user",
    json=test_user_current
)
if response2.status_code == 200:
    result2 = response2.json()
    print(f"✅ Status: {response2.status_code}")
    print(f"   User ID: {result2['user_id']}")
    print(f"   Source: {result2['source']}")
    print(f"   Same embedding: {result1['embedding'][:5] == result2['embedding'][:5]}")
else:
    print(f"❌ Error: {response2.status_code}")

# Test 3: Force regenerate
print("\n3. Testing /embed/user (Force regenerate)")
print("-"*60)
test_user_force = {**test_user_current, "force_regenerate": True}
response3 = requests.post(
    f"{BASE_URL}/embed/user",
    json=test_user_force
)
if response3.status_code == 200:
    result3 = response3.json()
    print(f"✅ Status: {response3.status_code}")
    print(f"   Source: {result3['source']}")
    print(f"   Forced regeneration: {result3['source'] == 'generated'}")
else:
    print(f"❌ Error: {response3.status_code}")

# Test 4: Batch user similarity
print("\n4. Testing /similarity/users (Batch similarity)")
print("-"*60)
similarity_request = {
    "current_user_id": test_user_current["user_id"],
    "current_user_data": {
        "major": test_user_current["major"],
        "faculty": test_user_current["faculty"],
        "courses": test_user_current["courses"],
        "skills": test_user_current["skills"],
        "bio": test_user_current["bio"]
    },
    "candidate_user_ids": [
        test_user_candidate1["user_id"],
        test_user_candidate2["user_id"]
    ],
    "candidate_users_data": [
        {
            "major": test_user_candidate1["major"],
            "faculty": test_user_candidate1["faculty"],
            "courses": test_user_candidate1["courses"],
            "skills": test_user_candidate1["skills"],
            "bio": test_user_candidate1["bio"]
        },
        {
            "major": test_user_candidate2["major"],
            "faculty": test_user_candidate2["faculty"],
            "courses": test_user_candidate2["courses"],
            "skills": test_user_candidate2["skills"],
            "bio": test_user_candidate2["bio"]
        }
    ]
}

response4 = requests.post(
    f"{BASE_URL}/similarity/users",
    json=similarity_request
)
if response4.status_code == 200:
    result4 = response4.json()
    print(f"✅ Status: {response4.status_code}")
    print(f"   Current User: {result4['current_user_id']}")
    print(f"   Similarities computed: {result4['count']}")
    for sim in result4['similarities']:
        print(f"   - {sim['user_id']}: {sim['similarity']:.4f}")
else:
    print(f"❌ Error: {response4.status_code}")
    print(f"   {response4.text}")

print("\n" + "="*60)
print("TESTING COMPLETE")
print("="*60)
