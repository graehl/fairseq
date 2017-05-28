#!/bin/bash
d=`dirname $0`
eval=$d/eval.sh
. $d/common.sh
echo1() {
    echo -n "$@"
}
echotab() {
    echo -n -e '\t' "$@"
}
fields() {
    local first="$1"
    echo1 $first
    shift
    for f in "$@"; do
        echotab "$f"
    done
    echo
}
megs() {
    du -BM "$@"
}
megs1() {
    megs "$1" | cut -f1
}
headers() {
    echo1 '#'
    fields "$@"
}
sortresults() {
    local results=$1
    sort -r -n -k 5 -k 8 "$results" > ${2:-$results.sorted}
}
main() {
    resultsheaders=${results:=results.tsv}.headers
    headers model beam lenpen BLEU METEOR TER Length > $resultsheaders
    cat $resultsheaders
    for f in $(ls -t $(find . -name model_best.th7)); do
        d=`dirname $f`
        dd=`basename $d`
        set -e
        beams="$*"
        [[ $beams ]] || beams="1 2 4 8 16"
        for beam in $beams; do
            if [[ $beam = 1 ]] ; then
                lens="4"
            else
                lens="2 4 8"
            fi
            for lenpen in $lens; do
                testf=$d/results.b$beam.lp$lenpen
                multscore=$testf.score.multeval
                quiet=1 rescore=1 beam=$beam lenpen=$lenpen testf=$testf $eval $d
                #2>/dev/null >/dev/null
                if [[ -s $multscore ]] ; then
                    echo1 $dd
                    echotab `megs1 $f`
                    echotab $beam
                    echotab $lenpen
                    echotab
                    grepmultbleu < $multscore
                    echo
                else
                    echo2 $multscore missing
                fi
            done
        done
        echo
    done | tee $results
    sortresults $results
}
main "$@" && exit 0 || exit 1
