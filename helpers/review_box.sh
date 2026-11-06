#!/usr/bin/env bash
# One round with the machine of the outside reviewers: what people saved there comes here, what they called completed
# ends, the drives given in Data › Reviewers go there without their places, the drives nobody holds leave it
# (nnslr review-box sync, nnslr_tools.review_box). The button of Data › Reviewers and a timer every 5 minutes both run
# this; a second one running at the same moment stops at once.
#
# usage: helpers/review_box.sh
# Writes the labels that come back and web/review_box in the data root; a new drive may take minutes to make its video,
# and sends about 3 GB to the machine.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec nice "$ROOT/.venv/bin/nnslr" review-box sync
