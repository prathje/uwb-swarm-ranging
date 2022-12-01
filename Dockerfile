FROM --platform=linux/amd64 zephyrprojectrtos/zephyr-build:latest

# Switch to root for installations
USER root

# Setup essentials
RUN apt-get update -y
RUN apt-get install -y git nano build-essential

# Install dependencies
RUN apt-get install -y gcc-multilib g++-multilib fftw3-dev

WORKDIR /

RUN mkdir -p /app
RUN mkdir -p /zephyr

RUN chown -Rf user:user /app
RUN chown -Rf user:user /zephyr

USER user

WORKDIR /zephyr

ENV ZEPHYR_BASE=/zephyr/zephyr

ENV ZEPHYR_REVISION=v2.7.2
RUN west init --mr $ZEPHYR_REVISION
RUN west update
RUN west zephyr-export

# Switch back to app directory
WORKDIR /app