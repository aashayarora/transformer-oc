#!/bin/bash
set -e

if [ ! -d "/home/transformer-oc/.git" ]; then
    git clone https://github.com/aashayarora/transformer-oc.git /home/transformer-oc
fi

cd /home/transformer-oc

exec "$@"