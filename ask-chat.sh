#!/bin/bash

curl -X 'POST' \
  'http://0.0.0.0:8000/chat?question=cuentame%20un%20chiste' \
  -H 'accept: application/json' \
  -d ''