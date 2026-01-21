#!/bin/bash

# Simplified Docker LaTeX Thesis Builder
# Two modes: with proxy or without proxy

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="latex-thesis-builder"
IMAGE_TAG="latest"
CONTAINER_NAME="thesis-builder"
OUTPUT_DIR="$(pwd)/output"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is available
check_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running or not accessible"
        exit 1
    fi
    
    print_success "Docker is available"
}

# Function to build Docker image
build_image() {
    local use_proxy=$1
    
    print_status "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
    
    # Prepare build command
    local build_cmd=("docker" "build")
    
    if [ "$use_proxy" = "true" ]; then
        print_status "Building with proxy settings"
        
        # Add proxy arguments if they exist
        if [ -n "$HTTP_PROXY" ] || [ -n "$http_proxy" ]; then
            local http_proxy_val="${HTTP_PROXY:-$http_proxy}"
            build_cmd+=("--build-arg" "HTTP_PROXY=$http_proxy_val" "--build-arg" "http_proxy=$http_proxy_val")
            print_status "Using HTTP proxy: $http_proxy_val"
        fi
        
        if [ -n "$HTTPS_PROXY" ] || [ -n "$https_proxy" ]; then
            local https_proxy_val="${HTTPS_PROXY:-$https_proxy}"
            build_cmd+=("--build-arg" "HTTPS_PROXY=$https_proxy_val" "--build-arg" "https_proxy=$https_proxy_val")
            print_status "Using HTTPS proxy: $https_proxy_val"
        fi
        
        if [ -n "$NO_PROXY" ] || [ -n "$no_proxy" ]; then
            local no_proxy_val="${NO_PROXY:-$no_proxy}"
            build_cmd+=("--build-arg" "NO_PROXY=$no_proxy_val" "--build-arg" "no_proxy=$no_proxy_val")
        fi
    else
        print_status "Building without proxy settings"
    fi
    
    # Add image tag and context
    build_cmd+=("-t" "$IMAGE_NAME:$IMAGE_TAG" ".")
    
    # Use legacy builder to avoid buildx issues
    export DOCKER_BUILDKIT=0
    
    # Execute build command
    print_status "Starting Docker build..."
    if "${build_cmd[@]}"; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to run the container
run_container() {
    local build_args="$*"
    
    print_status "Running LaTeX build in container"
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    # Remove existing container if it exists
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    
    # Run the container
    if docker run --name "$CONTAINER_NAME" \
        -v "$(pwd):/workspace" \
        -v "$OUTPUT_DIR:/workspace/output" \
        "$IMAGE_NAME:$IMAGE_TAG" \
        ./build_thesis.sh --output output $build_args; then
        
        print_success "LaTeX build completed successfully"
        
        # Check if PDF was generated
        if [ -f "$OUTPUT_DIR/main.pdf" ]; then
            local file_size=$(stat -c%s "$OUTPUT_DIR/main.pdf" 2>/dev/null || stat -f%z "$OUTPUT_DIR/main.pdf" 2>/dev/null)
            local file_size_mb=$((file_size / 1024 / 1024))
            print_success "PDF generated: $OUTPUT_DIR/main.pdf (${file_size_mb}MB)"
        else
            print_warning "PDF not found in output directory"
        fi
    else
        print_error "LaTeX build failed"
        
        # Show container logs for debugging
        print_status "Container logs:"
        docker logs "$CONTAINER_NAME"
        exit 1
    fi
}

# Function to run interactive shell in container
run_interactive() {
    print_status "Starting interactive shell in container"
    
    # Remove existing container if it exists
    docker rm -f "$CONTAINER_NAME-interactive" >/dev/null 2>&1 || true
    
    docker run -it --name "$CONTAINER_NAME-interactive" \
        -v "$(pwd):/workspace" \
        -v "$OUTPUT_DIR:/workspace/output" \
        "$IMAGE_NAME:$IMAGE_TAG" \
        /bin/bash
}

# Function to clean up Docker resources
cleanup() {
    print_status "Cleaning up Docker resources"
    
    # Remove containers
    docker rm -f "$CONTAINER_NAME" "$CONTAINER_NAME-interactive" >/dev/null 2>&1 || true
    
    # Optionally remove image
    if [ "$1" = "--image" ]; then
        docker rmi "$IMAGE_NAME:$IMAGE_TAG" >/dev/null 2>&1 || true
        print_success "Docker image removed"
    fi
    
    print_success "Cleanup completed"
}

# Function to show logs
show_logs() {
    if docker ps -a --format "table {{.Names}}" | grep -q "$CONTAINER_NAME"; then
        print_status "Showing container logs:"
        docker logs "$CONTAINER_NAME"
    else
        print_warning "Container $CONTAINER_NAME not found"
    fi
}

# Function to display help
show_help() {
    echo "Simplified Docker LaTeX Thesis Builder"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  with-proxy          Build and run with proxy settings"
    echo "  no-proxy            Build and run without proxy"
    echo "  shell               Start interactive shell in container (image must exist)"
    echo "  logs                Show container logs"
    echo "  clean               Clean up containers"
    echo "  clean --image       Clean up containers and image"
    echo "  help                Show this help message"
    echo ""
    echo "Build Options:"
    echo "  --clean             Clean auxiliary files before building"
    echo "  --fast              Fast build (skip bibliography)"
    echo ""
    echo "Environment Variables (for with-proxy mode):"
    echo "  HTTP_PROXY          HTTP proxy URL"
    echo "  HTTPS_PROXY         HTTPS proxy URL"
    echo "  NO_PROXY            Comma-separated list of hosts to exclude"
    echo ""
    echo "Examples:"
    echo "  $0 no-proxy                           # Build without proxy"
    echo "  $0 with-proxy                         # Build with proxy from environment"
    echo "  $0 no-proxy --clean                   # Clean build without proxy"
    echo "  $0 shell                              # Interactive development"
    echo "  HTTP_PROXY=http://proxy:8080 $0 with-proxy"
}

# Main execution
main() {
    local command="${1:-no-proxy}"
    shift || true
    
    case "$command" in
        with-proxy)
            check_docker
            build_image "true"
            run_container "$@"
            ;;
        no-proxy)
            check_docker
            build_image "false"
            run_container "$@"
            ;;
        shell)
            check_docker
            run_interactive
            ;;
        logs)
            show_logs
            ;;
        clean)
            cleanup "$1"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
