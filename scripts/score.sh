#!/bin/bash
#outputs: $G.score (fairseq) $G.multeval $G.score.detok.lcBLEU (xmt)
unbpe() {
    sed 's/@@ //g;s/__LW_SW__ //g;'
}
main() {
    set -e
    G=$1
    G=${G%.score}
    [[ -f $G ]]
    testscore=$G.score
    if [[ $redoall || ! $norescore || ! -s $testscore ]] ; then
        if [[ $redoall || $resplit || ! -s $G.sys.bpe || ! -s $G.ref.bpe ]] ; then
            grep ^H $G | cut -c3- | sort -n > $G.H
            grep ^T $G | cut -c3- | sort -n > $G.T
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
        dconf=.
        [[ -f $dconf/srctrglang.sh ]] || dconf=..
        if [[ -f $dconf/srctrglang.sh ]] ; then
            . $dconf/srctrglang.sh
            if [[ -f $dconf/xmtconfig.sh ]] ; then
                . $dconf/xmtconfig.sh
                xmtconfig
                score=$testscore.detok.lcBLEU
                if [[ $redoall || $redotokbleu || ! -s $score ]] ; then
                (set -e
                 for f in sys ref; do
                     if [[ $f = sys || ! -s $G.$f.detok.lc ]] ; then
                         xmtiopost $G.$f.bpe $G.$f.detok 2>/dev/null
                         xmtiolc $G.$f.detok $G.$f.detok.lc 2>/dev/null
                     fi
                 done
                 set -x
                 if [[ -f ${TESTDETOKLC:=test.0.eng} ]] ; then
                    bleuscore $G.sys.detok.lc $TESTDETOKLC > $score 2>/dev/null
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
            multeval $G.sys $G.ref > $multscore
        fi
        [[ $quiet ]] ||             echo `pwd`/$multscore
        [[ $quiet ]] ||             cat $multscore
    fi
}
main "$@" && exit 0 || exit 1
