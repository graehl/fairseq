#!/bin/bash
#outputs: $G.score (fairseq) $G.multeval $G.score.detok.lcBLEU (xmt)
unbpe() {
    sed 's/@@ //g;s/__LW_SW__ //g;'
}
cleanuplwat() {
    perl -pe 's/(^| )__LW_AT__( |$)/$1/g' "$@"
}
detoklwat() {
    perl -pe 's/ ?(?:__LW_AT__|\Q_-#-_\E) ?//go' "$@"
}
perlutf8() {
    local enutf8=${enutf8:-en_US.UTF-8}
    LC_ALL=$enutf8 perl "$@"
}
lcutf8() {
    perlutf8 -pe '$_=lc($_)' "$@"
}
main() {
    set -e
    G=$1
    G=${G%.score}
    [[ -f $G ]]
    testscore=$G.score
    if [[ $redoall || ! $norescore || ! -s $testscore ]] ; then
        if [[ $redoall || $resplit || $retokbleu || ! -s $G.sys.bpe || ! -s $G.ref.bpe ]] ; then
            grep ^H $G | cut -c3- | sort -n > $G.H
            grep ^T $G | cut -c3- | sort -n > $G.T
            [[ $quiet ]] || wc -l $G.H $G.T
            cut -f3- < $G.H | cleanuplwat > $G.sys.bpe
            cut -f2- < $G.T | cleanuplwat > $G.ref.bpe
        fi
        for f in sys ref; do
            unbpe < $G.$f.bpe > $G.$f
            [[ $quiet ]] || wc $G.$f $G.$f.bpe
        done
        fairseq score -sys $G.sys -ref $G.ref > $testscore
        [[ $quiet ]] || echo `pwd`/$testscore
        [[ $quiet ]] || cat $testscore
        . `dirname $0`/common.sh
        dconf=.
        [[ -f $dconf/srctrglang.sh ]] || dconf=..
        if [[ -f $dconf/srctrglang.sh ]] ; then
            . $dconf/srctrglang.sh
            if [[ -f $dconf/xmtconfig.sh ]] ; then
                . $dconf/xmtconfig.sh
                xmtconfig
                score=$testscore.detok.lcBLEU
                if [[ $redoall || $retokbleu || ! -s $score ]] ; then
                (set -e
                 for f in sys ref; do
                     if [[ $f = sys || ! -s $G.$f.detok.lc ]] ; then
                         set -x
                         #xmtiopost $G.$f.bpe $G.$f.detok >/dev/null 2>/dev/null
                         detoklwat $G.$f.bpe > $G.$f.detok
                         #xmtiolc $G.$f.detok $G.$f.detok.lc >/dev/null 2>/dev/null
                         lcutf8 $G.$f.detok > $G.$f.detok.lc
                     fi
                 done
                 set -x
                 if [[ -f ${detoklc:=${TESTDETOKLC:-test.0.eng}} ]] ; then
                    [[ $quiet ]] || echo "bleuscore $G.sys.detok.lc $detoklc | tee $score"
                    bleuscore $G.sys.detok.lc $detoklc > $score 2>/dev/null
                 else
                    bleuscore $G.sys.detok.lc $G.ref.detok.lc > $score 2>/dev/null
                 fi
                )
                fi
                [[ $quiet ]] ||         echo `pwd`/$score
                [[ $quiet ]] ||         cat $score
            fi
        fi
        multscore=$testscore.multeval
        if [[ $redoall || $rescore || ! -s $multscore ]] ; then
            multeval1ref $G.ref $G.sys > $multscore
        fi
        [[ $quiet ]] ||             echo `pwd`/$multscore `pwd`/$score
        [[ $quiet ]] ||             tail  $multscore $score
    fi
}
main "$@" && exit 0 || exit 1
