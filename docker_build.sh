#!/bin/bash

# Docker LaTeX Thesis Builder
# This script builds and runs the LaTeX thesis in a Docker container
# with support for proxy environments

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

# Function to build proxy arguments array
build_proxy_args() {
    local proxy_args=()
    
    # Check environment variables for proxy settings
    if [ -n "$HTTP_PROXY" ] || [ -n "$http_proxy" ]; then
        local http_proxy_val="${HTTP_PROXY:-$http_proxy}"
        proxy_args+=("--build-arg" "HTTP_PROXY=$http_proxy_val" "--build-arg" "http_proxy=$http_proxy_val")
    fi
    
    if [ -n "$HTTPS_PROXY" ] || [ -n "$https_proxy" ]; then
        local https_proxy_val="${HTTPS_PROXY:-$https_proxy}"
        proxy_args+=("--build-arg" "HTTPS_PROXY=$https_proxy_val" "--build-arg" "https_proxy=$https_proxy_val")
    fi
    
    if [ -n "$NO_PROXY" ] || [ -n "$no_proxy" ]; then
        local no_proxy_val="${NO_PROXY:-$no_proxy}"
        proxy_args+=("--build-arg" "NO_PROXY=$no_proxy_val" "--build-arg" "no_proxy=$no_proxy_val")
    fi
    
    # Return the array
    printf '%s\n' "${proxy_args[@]}"
}

# Function to build Docker image
build_image() {
    local use_proxy=$1
    local dockerfile=${2:-"Dockerfile"}
    
    print_status "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
    print_status "Using dockerfile: $dockerfile"
    
    # Prepare build command
    local build_cmd=("docker" "build")
    
    if [ "$use_proxy" = "true" ]; then
        # Check and display proxy settings
        if [ -n "$HTTP_PROXY" ] || [ -n "$http_proxy" ]; then
            local http_proxy_val="${HTTP_PROXY:-$http_proxy}"
            print_status "Detected HTTP proxy: $http_proxy_val"
        fi
        
        if [ -n "$HTTPS_PROXY" ] || [ -n "$https_proxy" ]; then
            local https_proxy_val="${HTTPS_PROXY:-$https_proxy}"
            print_status "Detected HTTPS proxy: $https_proxy_val"
        fi
        
        if [ -n "$NO_PROXY" ] || [ -n "$no_proxy" ]; then
            local no_proxy_val="${NO_PROXY:-$no_proxy}"
            print_status "Detected NO_PROXY: $no_proxy_val"
        fi
        
        # Get proxy arguments as array
        local proxy_args_array
        readarray -t proxy_args_array < <(build_proxy_args)
        
        if [ ${#proxy_args_array[@]} -gt 0 ]; then
            print_status "Building with proxy settings"
            # Add proxy arguments to build command
            build_cmd+=("${proxy_args_array[@]}")
        else
            print_warning "Proxy requested but no proxy environment variables found"
        fi
    else
        print_status "Building without proxy settings"
    fi
    
    # Add dockerfile and image tag and context
    build_cmd+=("-f" "$dockerfile" "-t" "$IMAGE_NAME:$IMAGE_TAG" ".")
    
    # Set environment variable to use legacy builder (avoid buildx issues)
    export DOCKER_BUILDKIT=0
    
    # Execute build command
    print_status "Running Docker build command..."
    if "${build_cmd[@]}"; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to run the container
run_container() {
    local build_args="$1"
    
    print_status "Running LaTeX build in container"
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    # Remove existing container if it exists
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    
    # Run the container
    local docker_cmd="docker run --name $CONTAINER_NAME"
    docker_cmd="$docker_cmd -v $(pwd):/workspace"
    docker_cmd="$docker_cmd -v $OUTPUT_DIR:/workspace/output"
    docker_cmd="$docker_cmd $IMAGE_NAME:$IMAGE_TAG"
    
    if [ -n "$build_args" ]; then
        docker_cmd="$docker_cmd $build_args"
    fi
    
    if eval "$docker_cmd"; then
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

# Function to watch mode
watch_mode() {
    print_status "Starting watch mode - building on file changes"
    print_status "Press Ctrl+C to stop"
    
    # Initial build
    run_container "--watch"
}

# Function to display help
show_help() {
    echo "Docker LaTeX Thesis Builder"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build               Build Docker image and run thesis compilation"
    echo "  build-proxy         Build with proxy settings from environment"
    echo "  build-no-proxy      Build without proxy settings"
    echo "  build-lite          Build lightweight version (faster, fewer packages)"
    echo "  build-lite-proxy    Build lightweight version with proxy"
    echo "  run                 Run thesis compilation (image must exist)"
    echo "  shell               Start interactive shell in container"
    echo "  watch               Watch mode - rebuild on file changes"
    echo "  logs                Show container logs"
    echo "  clean               Clean up containers"
    echo "  clean --image       Clean up containers and image"
    echo "  help                Show this help message"
    echo ""
    echo "Build Options (for build command):"
    echo "  --clean, -c         Clean auxiliary files before building"
    echo "  --fast, -f          Fast build (skip bibliography)"
    echo "  --output, -o DIR    Output directory (default: ./output)"
    echo ""
    echo "Environment Variables:"
    echo "  HTTP_PROXY          HTTP proxy URL"
    echo "  HTTPS_PROXY         HTTPS proxy URL"
    echo "  NO_PROXY            Comma-separated list of hosts to exclude"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Auto-detect proxy and build"
    echo "  $0 build-proxy              # Force use of proxy settings"
    echo "  $0 build-no-proxy           # Force no proxy"
    echo "  $0 build-lite               # Fast lightweight build"
    echo "  $0 run --clean              # Run with clean build"
    echo "  $0 shell                    # Interactive development"
    echo "  HTTP_PROXY=http://proxy:8080 $0 build-proxy"
}

# Main execution
main() {
    local command="${1:-build}"
    shift || true
    
    case "$command" in
        build)
            check_docker
            # Auto-detect proxy
            local use_proxy="auto"
            if [ -n "$HTTP_PROXY" ] || [ -n "$http_proxy" ] || [ -n "$HTTPS_PROXY" ] || [ -n "$https_proxy" ]; then
                use_proxy="true"
            else
                use_proxy="false"
            fi
            build_image "$use_proxy"
            run_container "$*"
            ;;
        build-proxy)
            check_docker
            build_image "true"
            run_container "$*"
            ;;
        build-no-proxy)
            check_docker
            build_image "false"
            run_container "$*"
            ;;
        build-lite)
            check_docker
            # Auto-detect proxy for lite build
            local use_proxy="auto"
            if [ -n "$HTTP_PROXY" ] || [ -n "$http_proxy" ] || [ -n "$HTTPS_PROXY" ] || [ -n "$https_proxy" ]; then
                use_proxy="true"
            else
                use_proxy="false"
            fi
            build_image "$use_proxy" "Dockerfile.lite"
            run_container "$*"
            ;;
        build-lite-proxy)
            check_docker
            build_image "true" "Dockerfile.lite"
            run_container "$*"
            ;;
        run)
            check_docker
            run_container "$*"
            ;;
        shell)
            check_docker
            run_interactive
            ;;
        watch)
            check_docker
            watch_mode
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
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
