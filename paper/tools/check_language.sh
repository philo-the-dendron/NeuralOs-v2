# Language rules of record (non-negotiable, from the session charter):
#   - "unattributed perturbation" is THE term for the adaptation's
#     downstream effects (adjudication 2026-08-21);
#   - no "steers"/"steering"/"steered" anywhere in our own voice;
#   - no "editing"/"edit"/"edited"/"editor" describing OUR work;
#   - no "verified-attribution" (cut with C1's scope triple).
# Sanctioned exceptions, line-marked with %%ALLOW%% :
#   (1) the verbatim ISC-82 quotation in sections/adjudication.tex;
#   (2) the literature's own names in sections/related.tex
#       (e.g. "knowledge editing", AlphaEdit) — field terms, cited.
# Exit 0 = clean; anything else prints the offenders and exits 1.
cd "$(dirname "$0")/.."
PATTERN='\<(steers?|steering|steered|editing|edited|editor|edit|verified-attribution)\>'
STATUS=0
for f in sections/*.tex main.tex; do
  # strip %%ALLOW%%-marked lines before matching
  offenders=$(grep -v '%%ALLOW%%' "$f" | grep -inE "$PATTERN" /dev/stdin \
    | sed "s|/dev/stdin|$f|")
  if [ -n "$offenders" ]; then
    echo "LANGUAGE VIOLATION in $f:"
    echo "$offenders"
    STATUS=1
  fi
done
if [ $STATUS -eq 0 ]; then echo "language gate: clean"; fi
exit $STATUS
