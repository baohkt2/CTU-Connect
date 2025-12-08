# Recommend Service - Documentation Index

## 📖 Quick Navigation

### 🚀 Getting Started
- [README-OPTIMIZED.md](./README-OPTIMIZED.md) - Main README với quick start
- [QUICK-TEST-REFERENCE.md](./QUICK-TEST-REFERENCE.md) - Quick test commands

### 🏗️ Architecture
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Original architecture document
- [ARCHITECTURE-OPTIMIZED.md](./ARCHITECTURE-OPTIMIZED.md) - Optimized architecture (v2.0)

### 🔄 Recent Changes
- [OPTIMIZATION-SUMMARY.md](./OPTIMIZATION-SUMMARY.md) - Tóm tắt optimization (v2.0)
- [CHANGES-LOG.md](./CHANGES-LOG.md) - Detailed changelog
- [REFACTORING-PLAN.md](./REFACTORING-PLAN.md) - Refactoring plan

### 📡 API Documentation
- [API-MIGRATION-GUIDE.md](./API-MIGRATION-GUIDE.md) - API migration guide for frontend
- [API-FLOW-DOCUMENTATION.md](./API-FLOW-DOCUMENTATION.md) - API flow documentation

### 🧪 Testing
- **[TESTING-COMPLETE-SUMMARY.md](./TESTING-COMPLETE-SUMMARY.md)** - ⭐ Start here for testing
- [TEST-SCRIPTS-GUIDE.md](./TEST-SCRIPTS-GUIDE.md) - Detailed test scripts guide
- [README-TESTING.md](./README-TESTING.md) - Complete testing documentation
- [QUICK-TEST-REFERENCE.md](./QUICK-TEST-REFERENCE.md) - Quick reference card

### 📝 Integration & Setup
- [README-INTEGRATION.md](./README-INTEGRATION.md) - Integration guide
- [QUICK-START.md](./QUICK-START.md) - Quick start guide

## 🎯 Common Tasks

### I want to...

#### Start the services
→ See [README-OPTIMIZED.md](./README-OPTIMIZED.md#quick-start)

#### Test the APIs
→ Run `./run-all-tests.ps1` or see [TESTING-COMPLETE-SUMMARY.md](./TESTING-COMPLETE-SUMMARY.md)

#### Understand the architecture
→ See [ARCHITECTURE-OPTIMIZED.md](./ARCHITECTURE-OPTIMIZED.md)

#### Migrate frontend to new APIs
→ See [API-MIGRATION-GUIDE.md](./API-MIGRATION-GUIDE.md)

#### See what changed recently
→ See [OPTIMIZATION-SUMMARY.md](./OPTIMIZATION-SUMMARY.md)

#### Debug issues
→ Check service logs and [README-TESTING.md](./README-TESTING.md#troubleshooting)

## 📂 File Organization

### Documentation (You are here)
```
recommend-service/
├── INDEX.md                          # This file - navigation
├── README-OPTIMIZED.md               # Main README
├── QUICK-TEST-REFERENCE.md           # Quick commands
├── ARCHITECTURE-OPTIMIZED.md         # System architecture
├── TESTING-COMPLETE-SUMMARY.md       # Testing summary ⭐
└── ... (other docs)
```

### Test Scripts
```
recommend-service/
├── test-seed-data.sql                # Seed test data
├── test-api-quick.ps1                # Quick test
├── test-api-comprehensive.ps1        # Full test
├── clean-test-data.ps1               # Cleanup
└── run-all-tests.ps1                 # Automated runner
```

### Source Code
```
recommend-service/
├── java-api/                         # Java Spring Boot service
│   └── src/main/java/.../
│       ├── controller/
│       │   └── RecommendationController.java  (UNIFIED)
│       └── service/
│           └── HybridRecommendationService.java
│
└── python-model/                     # Python AI service
    ├── server.py                     (UNIFIED ENTRY POINT)
    ├── inference.py
    └── api/routes.py
```

## 🔍 Search by Topic

### Architecture
- Original: [ARCHITECTURE.md](./ARCHITECTURE.md)
- Optimized: [ARCHITECTURE-OPTIMIZED.md](./ARCHITECTURE-OPTIMIZED.md)
- Changes: [OPTIMIZATION-SUMMARY.md](./OPTIMIZATION-SUMMARY.md)

### API Endpoints
- New endpoints: [API-MIGRATION-GUIDE.md](./API-MIGRATION-GUIDE.md)
- API flow: [API-FLOW-DOCUMENTATION.md](./API-FLOW-DOCUMENTATION.md)

### Testing
- Quick start: [TESTING-COMPLETE-SUMMARY.md](./TESTING-COMPLETE-SUMMARY.md)
- Full guide: [TEST-SCRIPTS-GUIDE.md](./TEST-SCRIPTS-GUIDE.md)
- Reference: [QUICK-TEST-REFERENCE.md](./QUICK-TEST-REFERENCE.md)

### Development
- Quick start: [QUICK-START.md](./QUICK-START.md)
- Integration: [README-INTEGRATION.md](./README-INTEGRATION.md)

## 📊 Documentation Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| Architecture | ✅ Updated | 2024-12-08 |
| API Guide | ✅ Updated | 2024-12-08 |
| Testing Suite | ✅ Complete | 2024-12-08 |
| Quick Start | ✅ Updated | 2024-12-08 |

## 🎯 Recommended Reading Order

### For New Developers
1. [README-OPTIMIZED.md](./README-OPTIMIZED.md) - Understand the system
2. [ARCHITECTURE-OPTIMIZED.md](./ARCHITECTURE-OPTIMIZED.md) - Learn architecture
3. [QUICK-TEST-REFERENCE.md](./QUICK-TEST-REFERENCE.md) - Test it
4. [API-MIGRATION-GUIDE.md](./API-MIGRATION-GUIDE.md) - Use the APIs

### For Frontend Developers
1. [API-MIGRATION-GUIDE.md](./API-MIGRATION-GUIDE.md) - New API endpoints
2. [QUICK-TEST-REFERENCE.md](./QUICK-TEST-REFERENCE.md) - Test APIs manually

### For DevOps/QA
1. [TESTING-COMPLETE-SUMMARY.md](./TESTING-COMPLETE-SUMMARY.md) - Testing overview
2. [TEST-SCRIPTS-GUIDE.md](./TEST-SCRIPTS-GUIDE.md) - Run tests
3. [README-TESTING.md](./README-TESTING.md) - Troubleshooting

### For Project Managers
1. [OPTIMIZATION-SUMMARY.md](./OPTIMIZATION-SUMMARY.md) - What changed
2. [CHANGES-LOG.md](./CHANGES-LOG.md) - Detailed changes

## 🔗 External Resources

- **Java API**: Port 8095
- **Python Service**: Port 8097
- **PostgreSQL**: Port 5435
- **Redis**: Port 6380
- **Neo4j**: Port 7687

## 📞 Support

For issues or questions:
1. Check relevant documentation above
2. Check service logs in `logs/` directories
3. Refer to troubleshooting sections

---

**Index Version**: 1.0.0
**Last Updated**: 2024-12-08
**Maintainer**: CTU Connect Team
