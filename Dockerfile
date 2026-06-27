# LaTeX Thesis Build Environment
# This Dockerfile creates an Ubuntu environment with full TeXLive installation
# for building LaTeX documents with proxy support

FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Build arguments for proxy configuration
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

# Set proxy environment variables if provided
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}

# Configure apt proxy if HTTP_PROXY is set
RUN if [ -n "$HTTP_PROXY" ]; then \
        echo "Acquire::http::Proxy \"$HTTP_PROXY\";" > /etc/apt/apt.conf.d/01proxy && \
        echo "Acquire::https::Proxy \"$HTTPS_PROXY\";" >> /etc/apt/apt.conf.d/01proxy; \
    fi

# Update package list and install essential packages first
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    nano \
    unzip \
    ca-certificates \
    build-essential \
    make \
    python3 \
    python3-pip \
    inotify-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install TeXLive in a separate layer (this is the largest part)
RUN apt-get update && apt-get install -y \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install additional TeXLive packages
RUN apt-get update && apt-get install -y \
    texlive-science \
    texlive-bibtex-extra \
    biber \
    texlive-lang-english \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install image processing tools
RUN apt-get update && apt-get install -y \
    imagemagick \
    ghostscript \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/apt/apt.conf.d/01proxy

# Create a non-root user for building
RUN useradd -m -s /bin/bash builder && \
    echo "builder ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /workspace

# Copy only the build script first to leverage Docker cache
COPY build_thesis.sh /workspace/
RUN chmod +x /workspace/build_thesis.sh

# Copy the rest of the source code
COPY . /workspace/

# Change ownership to builder user
RUN chown -R builder:builder /workspace

# Switch to builder user
USER builder

# Create output directory
RUN mkdir -p /workspace/output

# Set default command to build the thesis
CMD ["./build_thesis.sh", "--output", "output"]

# Health check to verify LaTeX installation
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pdflatex --version || exit 1

# Labels for metadata
LABEL maintainer="LaTeX Thesis Builder"
LABEL description="Ubuntu environment with essential TeXLive for building LaTeX documents"
LABEL version="1.0"
