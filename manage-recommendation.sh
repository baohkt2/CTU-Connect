#!/bin/bash
# Script quản lý recommendation-service với volume caching

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== CTU Connect Recommendation Service Manager ===${NC}"

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     - Start recommendation-service và dependencies"
    echo "  stop      - Stop recommendation-service"
    echo "  restart   - Restart recommendation-service"
    echo "  rebuild   - Rebuild image (giữ lại thư viện trong volume)"
    echo "  clean     - Xóa containers và rebuild từ đầu (giữ volumes)"
    echo "  reset     - Xóa tất cả (bao gồm volumes) và rebuild"
    echo "  logs      - Xem logs của recommendation-service"
    echo "  status    - Kiểm tra status của services"
    echo "  shell     - Truy cập shell của container"
    echo ""
}

# Function to start services
start_services() {
    echo -e "${YELLOW}🚀 Starting recommendation-service và dependencies...${NC}"
    docker-compose up -d eureka-server kafka redis recommendation_db
    echo -e "${YELLOW}⏳ Chờ dependencies khởi động...${NC}"
    sleep 10
    docker-compose up -d recommendation-service
    echo -e "${GREEN}✅ Services started successfully!${NC}"
}

# Function to stop services
stop_services() {
    echo -e "${YELLOW}🛑 Stopping recommendation-service...${NC}"
    docker-compose stop recommendation-service
    echo -e "${GREEN}✅ Service stopped!${NC}"
}

# Function to restart service
restart_service() {
    echo -e "${YELLOW}🔄 Restarting recommendation-service...${NC}"
    docker-compose restart recommendation-service
    echo -e "${GREEN}✅ Service restarted!${NC}"
}

# Function to rebuild (keep volumes)
rebuild_service() {
    echo -e "${YELLOW}🔨 Rebuilding recommendation-service (thư viện sẽ được giữ lại)...${NC}"
    docker-compose stop recommendation-service
    docker-compose rm -f recommendation-service
    docker-compose build --no-cache recommendation-service
    docker-compose up -d recommendation-service
    echo -e "${GREEN}✅ Service rebuilt successfully! Thư viện Python vẫn được giữ trong volume.${NC}"
}

# Function to clean and rebuild
clean_rebuild() {
    echo -e "${YELLOW}🧹 Cleaning containers và rebuilding (giữ volumes)...${NC}"
    docker-compose down recommendation-service
    docker-compose build --no-cache recommendation-service
    docker-compose up -d eureka-server kafka redis recommendation_db
    sleep 5
    docker-compose up -d recommendation-service
    echo -e "${GREEN}✅ Clean rebuild completed!${NC}"
}

# Function to reset everything
reset_all() {
    echo -e "${RED}⚠️  CẢNH BÁO: Sẽ xóa TẤT CẢ dữ liệu và thư viện!${NC}"
    read -p "Bạn có chắc chắn? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}🗑️  Resetting everything...${NC}"
        docker-compose down -v recommendation-service
        docker volume rm ctu-connect_recommendation-venv ctu-connect_recommendation-app-data ctu-connect_recommendation-cache 2>/dev/null || true
        docker-compose build --no-cache recommendation-service
        start_services
        echo -e "${GREEN}✅ Full reset completed!${NC}"
    else
        echo -e "${BLUE}Operation cancelled.${NC}"
    fi
}

# Function to show logs
show_logs() {
    echo -e "${BLUE}📋 Recommendation-service logs:${NC}"
    docker-compose logs -f recommendation-service
}

# Function to show status
show_status() {
    echo -e "${BLUE}📊 Services Status:${NC}"
    echo ""
    echo -e "${YELLOW}Recommendation Service:${NC}"
    docker-compose ps recommendation-service
    echo ""
    echo -e "${YELLOW}Dependencies:${NC}"
    docker-compose ps eureka-server kafka redis recommendation_db
    echo ""
    echo -e "${YELLOW}Volumes:${NC}"
    docker volume ls | grep -E "(recommendation-venv|recommendation-app-data|recommendation-cache)"
    echo ""
    echo -e "${YELLOW}Health Check:${NC}"
    curl -s http://localhost:8086/health | jq . 2>/dev/null || curl -s http://localhost:8086/health
}

# Function to access shell
access_shell() {
    echo -e "${BLUE}🐚 Truy cập shell của recommendation-service container...${NC}"
    docker-compose exec recommendation-service /bin/bash
}

# Main script logic
case "${1:-}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_service
        ;;
    rebuild)
        rebuild_service
        ;;
    clean)
        clean_rebuild
        ;;
    reset)
        reset_all
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    shell)
        access_shell
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
