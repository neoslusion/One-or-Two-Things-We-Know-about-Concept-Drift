#!/bin/bash

# Terminate all docker containers
docker compose down

# Remove all docker containers
docker rm $(docker ps -a -q)
