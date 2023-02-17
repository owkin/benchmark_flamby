#!/usr/bin/bash
benchopt run --max-runs 12 --timeout 360000000000 -s FederatedAveraging
benchopt run --max-runs 12 --timeout 360000000000 -s Cyclic
benchopt run --max-runs 12 --timeout 360000000000 -s FedProx
benchopt run --max-runs 12 --timeout 360000000000 -s Scaffold
benchopt run --max-runs 12 --timeout 360000000000 -s FedAdam
benchopt run --max-runs 12 --timeout 360000000000 -s FedAdagrad
benchopt run --max-runs 12 --timeout 360000000000 -s FedYogi