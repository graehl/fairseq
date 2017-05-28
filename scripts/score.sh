#!/bin/bash
#outputs: $G.score (fairseq) $G.multeval $G.score.detok.lcBLEU (xmt)
unbpe() {
    sed 's/@@ //g;s/__LW_SW__ //g;'
}
main() {
    set -e
    G=$1
    [[ -f $G ]]
    testscore=$G.score
    if [[ $redoall || ! $norescore || ! -s $testscore ]] ; then
        if [[ $redoall || $resplit || ! -s $G.sys.bpe || ! -s $G.ref.bpe ]] ; then
            grep ^H $G > $G.H
            grep ^T $G > $G.T
            [[ $quiet ]] || wc -l $G.H $G.T
            cut -f3- < $G.H > $G.sys.bpe
            cut -f2- < $G.T > $G.ref.bpe
        fi
        for f in sys ref; do
            unbpe < $G.$f.bpe > $G.$f
            [[ $quiet ]] || wc $G.$f $G.$f.bpe
        done
        fairseq score -sys $G.sys -ref $G.ref > $testscore
        [[ $quiet ]] || echo `pwd`/$testscore
        [[ $quiet ]] || cat $testscore
        . `dirname $0`/common.sh
        if [[ -f srctrglang.sh ]] ; then
            . srctrglang.sh
            if [[ -f xmtconfig.sh ]] ; then
                . xmtconfig.sh
                xmtconfig
                score=$testscore.detok.lcBLEU
                if [[ $redoall || $redetokbleu || ! -s $score ]] ; then
                (set -e
                 set -x
                 for f in sys ref; do
                     xmtiopost $G.$f.bpe $G.$f.detok 2>/dev/null
                     xmtiolc $G.$f.detok $G.$f.detok.lc 2>/dev/null
                 done
                 bleuscore $G.sys.detok.lc $G.ref.detok.lc >$score 2>/dev/null
                )
                fi
                [[ $quiet ]] ||         echo `pwd`/$score
                [[ $quiet ]] ||         cat $score
            fi
        fi
        multscore=$testscore.multeval
        if [[ $redoall || $rescore || ! -s $multscore ]] ; then
            multeval $G.sys $G.ref > $multscore
        fi
        [[ $quiet ]] ||             echo `pwd`/$multscore
        [[ $quiet ]] ||             cat $multscore
    fi
}
main "$@" && exit 0 || exit 1
