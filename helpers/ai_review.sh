#!/usr/bin/env bash
# Reviews the frames of a route that nobody confirmed yet with an AI provider of Settings (nnslr ai-review): the
# "Review with AI" button of the route list starts this. Frames a person confirmed are never touched.
#
# Without --save it is a trial: nothing goes into the reviews, the suggestions go to data/ai_reviews/<route>/.
# With --save every frame the AI judged completely is saved as an AI review (reviewer ai/<provider>/<model>). An
# AI save is final: nobody approves it again, and the next import ("Import reviews", helpers/3_train.sh) puts it into
# the training data.
#
# usage: helpers/ai_review.sh ROUTE [--provider=ID] [--save] [--limit=N] [--scope=lights]
#   (default provider: the reviewer of Settings; --scope=lights: only the frames that come to a traffic light of the
#   map, each still reviewed whole)
# The provider's key is read from its file by the command: it is never an argument, an environment variable or a
# line of this log. No GPU here: the frames go to the provider's server.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ "$#" -lt 1 ]; then
    echo "usage: helpers/ai_review.sh ROUTE [--provider=ID] [--save] [--limit=N] [--scope=lights]" >&2
    exit 2
fi
route="$1"
shift
save=no
for argument in "$@"; do
    if [ "$argument" = "--save" ]; then
        save=yes
    fi
done
if [ "$save" = yes ]; then
    echo "Reviewing the route $route with the AI, saving its reviews..."
else
    echo "Reviewing the route $route with the AI, a trial: nothing is saved..."
fi
"$ROOT/.venv/bin/nnslr" ai-review "$route" "$@" > /dev/null
if [ "$save" = yes ]; then
    echo "Done: the AI's reviews are saved; the next import takes them into training."
else
    echo "Done: a trial, nothing saved; the suggestions are in data/ai_reviews/$route/."
fi
